import torch
import torch.nn as nn


class MoD(nn.Module):
    """The Mixtures of Depth Block that dynamically which tokens to process in a block.
    Wraps around decoder block to allow for token dropping.
    """

    def __init__(self, config, block):
        super().__init__()
        self.block = block  # block is att + mlp
        self.mod_router = nn.Linear(config.d_model, 1, bias=False)
        # capacity factor is between [0,1),
        # where for 1 we recover a vanilla transformer
        # (i.e. all tokens passed through block)
        self.capacity_factor = config.mod_capacity_factor
        self.top_k = int(self.capacity_factor * config.max_sequence_length)

        # mlp router and bce loss are used for inference
        self.mlp_router = nn.Linear(config.d_model, 1, bias=False)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def reset_parameters(self):
        self.block.reset_parameters()

    def set_activation_checkpointing(self, strategy):
        self.block.set_activation_checkpointing(strategy)

    def get_aux_loss(self, x, targets):
        # with a stop-loss
        B, T, C = x.shape
        mlp_router_logits = self.mlp_router(x.detach().view(B*T, -1))
        return self.bce_loss(mlp_router_logits.view(-1), targets)

    def forward(self, x, **kwargs):
        # [batch_size, sequence_length, n_embd]
        B, T, C = x.shape
        # inference time optimization: sequence length can
        # be smaller than seq len during training
        top_k = min(self.top_k, int(self.capacity_factor * T))

        """STEP 1: get logits and top_k tokens"""
        # [batch_size, sequence_length, 1]
        router_logits = self.mod_router(x)
        # weights and selected tokens: [batch_size, top_k, 1]
        weights, selected_tokens = torch.topk(router_logits, top_k, dim=1, sorted=False)

        # 0, if not in topk tokens, 1 else
        mlp_targets = torch.zeros_like(router_logits).view(-1)
        mlp_targets[selected_tokens.view(-1)] = 1.0
        aux_loss = self.get_aux_loss(x, mlp_targets)

        # IMPORTANT: need to sort indices to keep causal order for those tokens that
        # are processed in a block
        selected_tokens, index = torch.sort(selected_tokens, dim=1)
        weights = torch.gather(weights, dim=1, index=index)

        """STEP 2: expand indices to process batches with _reduced_ seqlen"""
        # We need to expand indices' dimensions from
        # [batch_size, top_k, 1] to [batch_size, top_k, n_embd] for gathering
        indices_expanded = selected_tokens.expand(-1, -1, C)
        # [batch_size, top_k, n_embd]
        top_k_tokens = torch.gather(x, 1, indices_expanded)
        top_k_tokens_processed, cache = self.block(top_k_tokens, **kwargs)

        """STEP 3: combine results"""
        x = torch.scatter_add(
            x,
            dim=1,
            index=indices_expanded,
            src=top_k_tokens_processed * weights,
        )

        return x, cache, aux_loss
