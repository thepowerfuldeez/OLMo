from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class top_k_gating(nn.Module):
    def __init__(
            self,
            input_size,
            num_experts,
            top_k,
    ):
        """
        Initialize the top-k gating mechanism.

        Args:
            input_size (int): Size of the input.
            num_experts (int): Number of experts.
            top_k (int): Number of top experts to select.
            acc_aux_loss (bool): Whether to accumulate auxiliary loss statistics.
            dropout (float): Dropout rate for gating network.
            hidden_size (int): Hidden size of the gating network.
            sample_topk (int): Number of top-k experts to sample during training.
            aux_loss (str): Type of auxiliary loss ('mi' or 'switch').
            gate_type (str): Type of gating mechanism ('mlp', 'linear', or 'gmm').
        """
        super().__init__()

        self.num_experts = num_experts
        self.input_size = input_size
        assert top_k <= num_experts
        self.top_k = top_k

        self.layer = nn.Linear(input_size, num_experts, bias=False)

    def extra_repr(self):
        """
        Return extra representation string for the module.
        """
        return "k={}, num_experts={}".format(self.top_k, self.num_experts)

    def compute_aux_loss(self, probs, logits, gates, eps=1e-5):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.size(0)
        probs = probs.sum(0)
        freq = (gates > 0).float().sum(0)
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1) + eps) ** 2).sum()

        switchloss = self.num_experts * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss
        # print("lsesq", lsesq, "switchloss", switchloss, "zloss", zloss)

        return loss

    def forward(self, x):
        """
        Compute the top-k gating for the input.

        See paper: https://arxiv.org/abs/1701.06538.

        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, input_size].
            skip_mask (torch.Tensor): Skip mask tensor (binary) with the same shape as `x`.
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float

        Returns:
            torch.Tensor: Top-k indices.
            torch.Tensor: Top-k gating values.
            torch.Tensor: Probability values for each expert.
            gates: a Tensor with shape [batch_size, num_experts]
            load: a Tensor with shape [num_experts]
        """

        logits = self.layer(x).float()
        top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)
        top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(x)

        if self.training:
            probs = torch.softmax(logits, dim=1)
            zeros = torch.zeros_like(probs)
            zeros = zeros.to(top_k_gates.dtype)  # Convert zeros to match top_k_gates dtype
            gates = zeros.scatter(1, top_k_indices, top_k_gates)
            self.loss = self.compute_aux_loss(probs, logits, gates)
        else:
            self.loss = 0

        return top_k_indices, top_k_gates


@torch.jit.script
def compute_gating(k: int, num_experts: int, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """
    Compute gating values for the mixture of experts based on probabilities and top-k indices.

    Args:
        k (int): Number of experts to select.
        num_experts (int): Total number of experts.
        top_k_gates (torch.Tensor): Gating values for top-k experts (batch_size x k).
        top_k_indices (torch.Tensor): Indices of top-k experts (batch_size x k).

    Returns:
        torch.Tensor: Batch-level gating values.
        torch.Tensor: Batch-level expert indices.
        torch.Tensor: Expert size for each expert.
        torch.Tensor: Sorted indices of top-k experts.
    """
    zeros = torch.zeros([top_k_gates.size(0), num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device)
    gates = zeros.scatter(1, top_k_indices, 1)
    expert_size = gates.long().sum(0)
    top_k_gates = top_k_gates.flatten()
    top_k_experts = top_k_indices.flatten()
    _, index_sorted_experts = top_k_experts.sort(0)
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates[index_sorted_experts]
    return batch_gates, batch_index, expert_size, index_sorted_experts


class ParallelExperts(nn.Module):
    def __init__(self, num_experts, input_size, output_size) -> None:
        """
        Initialize the ParallelExperts module.

        Args:
            num_experts (int): Number of experts.
            input_size (int): Size of the input.
            output_size (int): Size of the output.
            bias (bool): Whether to include bias terms.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, output_size, input_size))
        self.reset_parameters()
        self.num_experts = num_experts
        self.input_size = input_size
        self.output_size = output_size

    def extra_repr(self):
        return "num_experts={}, input_size={}, output_size={}".format(
            self.num_experts, self.input_size, self.output_size
        )

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the model.
        """
        nn.init.uniform_(self.weight, -1.0 / self.weight.size(1), 1.0 / self.weight.size(1))

    def forward(self, inputs, expert_size):
        """
        Forward pass of the ParallelExperts module.

        Args:
            inputs (Tensor): Input tensor.
            expert_size: Expert size information.

        Returns:
            Tensor: Output tensor.
        """
        input_list = inputs.split(expert_size, dim=0)
        output_list = []
        for i in range(self.num_experts):
            output_list.append(F.linear(input_list[i], self.weight[i]))
        # expert_size * num_experts x output_size
        results = torch.cat(output_list, dim=0)
        return results


class MoE(nn.Module):
    """
    A Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.

    Args:
        input_size: integer - size of the input
        hidden_size: integer - size of the expert's hidden layer
        num_experts: an integer - number of experts
        top_k: an integer - how many experts to use for each batch element
        bias: a boolean - whether to include bias in linear layers
        activation: an activation function to apply to expert's outputs
        glu: an boolean - whether to use GLU activation
    """

    def __init__(
            self,
            input_size,
            hidden_size,
            num_experts,
            top_k,
            bias=True,
            activation=None,
            activation_output_multiplier=True,
    ):
        super(MoE, self).__init__()

        self.num_experts = num_experts
        self.in_features = input_size
        self.input_size = input_size
        self.out_features = hidden_size
        self.hidden_size = hidden_size
        # self.glu = glu
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(input_size))
            torch.nn.init.zeros_(self.bias)
        else:
            self.bias = None

        # This is full MLP

        self.input_linear = ParallelExperts(num_experts, input_size, hidden_size)
        self.output_linear = ParallelExperts(num_experts, int(hidden_size * activation_output_multiplier), input_size)

        self.top_k = min(top_k, self.num_experts)
        self.activation = activation

        self.router = top_k_gating(
            input_size=input_size,
            num_experts=num_experts,
            top_k=top_k,
        )

    def extra_repr(self):
        return "k={}, e={}".format(self.top_k, self.num_experts)

    def get_aux_loss_and_clear(self):
        """
        Get the accumulated auxiliary loss and clear it.

        Returns:
            float: Accumulated auxiliary loss.
        """

        return self.gate.get_aux_loss_and_clear()

    def compute_gate(self, x):
        top_k_indices, self.top_k_gates = self.router(x)

        self.batch_gates, self.batch_index, expert_size, self.index_sorted_experts = compute_gating(
            self.top_k, self.num_experts, self.top_k_gates, top_k_indices
        )
        self.expert_size = expert_size.tolist()

        return self.router.loss

    def batch_forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        h = self.input_linear(expert_inputs, self.expert_size)

        if self.activation is not None:
            h = self.activation(h)
            assert torch.is_tensor(h)
        # if self.glu:
        #     h, g = h.chunk(2, dim=-1)
        #     h = self.activation(h) * g
        # else:
        #     h = self.activation(h)
        expert_outputs = self.output_linear(h, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        # return the same shape as needed
        y = expert_outputs.view(bsz, length, -1, emb_size).sum(-2)

        # zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        # y = zeros.index_add(0, self.batch_index, expert_outputs)
        # y = y.view(bsz, length, self.input_size)
        # if self.bias is not None:
        #     y = y + self.bias
        return y, loss

    def single_forward(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        top_k_indices, top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = top_k_indices[0, i]

            h = F.linear(x, self.input_linear.weight[expert_idx])
            if self.glu:
                h, g = h.chunk(2, dim=-1)
                h = self.activation(h) * g
            else:
                h = self.activation(h)
            y = F.linear(h, self.output_linear.weight[expert_idx]) * top_k_gates[0, i]

            y_list.append(y)

        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y, loss

    def forward(self, x):
        """
        Forward pass of the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length == 1:
            return self.single_forward(x)
        else:
            return self.batch_forward(x)

    def single_map(self, x):
        bsz, length, emb_size = x.size()

        x = x.reshape(1, self.input_size)
        self.top_k_indices, self.top_k_gates = self.router(x)
        loss = self.router.loss

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x, self.input_linear.weight[expert_idx])
            y_list.append(y)
        y = torch.cat(y_list, dim=0)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def batch_map(self, x):
        """

        Args:
            x: tensor shape [batch_size, input_size]
            train: a boolean scalar.
            loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
            y: a tensor with shape [batch_size, output_size].
            extra_training_loss: a scalar.  This should be added into the overall
            training loss of the model.  The backpropagation of this loss
            encourages all experts to be approximately equally used across a batch.
        """
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.
            skip_mask (Tensor): Skip mask tensor.
            sample_topk (int): Number of experts to sample during training.
            return_indices (bool): Whether to return expert indices.

        Returns:
            Tensor: Output tensor.
            float: Gating loss.
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)
        loss = self.compute_gate(x)

        expert_inputs = x[self.batch_index]
        expert_outputs = self.input_linear(expert_inputs, self.expert_size)

        zeros = torch.zeros(
            (bsz * length * self.top_k, self.hidden_size), dtype=expert_outputs.dtype, device=expert_outputs.device
        )
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.top_k, -1)
        return y, loss

    def map(self, x):
        """
        Map input through the mixture of experts layer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        bsz, length, emb_size = x.size()
        if bsz * length == 1:
            return self.single_map(x)
        else:
            return self.batch_map(x)

    def single_reduce(self, x):
        bsz, length, k, emb_size = x.size()

        x = x.reshape(k, emb_size)

        y_list = []
        for i in range(self.top_k):
            expert_idx = self.top_k_indices[0, i]
            y = F.linear(x[i], self.output_linear.weight[expert_idx]) * self.top_k_gates[0, i]
            y_list.append(y)
        y = sum(y_list)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def batch_reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.
            multiply_by_gates (bool): Whether to multiply outputs by gating values.

        Returns:
            Tensor: Reduced output tensor.
        """

        bsz, length, k, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_linear(expert_inputs, self.expert_size)

        expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        if self.bias is not None:
            y = y + self.bias
        return y

    def reduce(self, x):
        """
        Reduce the mapped output.

        Args:
            x (Tensor): Mapped output tensor.

        Returns:
            Tensor: Reduced output tensor.
        """
        bsz, length, k, emb_size = x.size()
        if bsz * length == 1:
            return self.single_reduce(x)
        else:
            return self.batch_reduce(x)
