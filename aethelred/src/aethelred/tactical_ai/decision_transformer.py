"""Decision Transformer for tactical decision-making."""

from __future__ import annotations

import torch
from torch import nn

from aethelred.config.settings import DecisionTransformerConfig
from aethelred.tactical_ai.action_head import ActionHead


class DecisionTransformer(nn.Module):
    """
    Decision Transformer for tactical AI.

    Sequence format per timestep: [return_token, state_token, action_token]
    Full sequence: [R_0, S_0, A_0, R_1, S_1, A_1, ..., R_t, S_t, ?]
    The model predicts A_t given the history.
    """

    def __init__(self, config: DecisionTransformerConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        # Embedding layers for each token type
        self.state_embed = nn.Linear(config.state_embed_dim, hidden_dim)
        self.action_embed = nn.Sequential(
            nn.Linear(config.action_embed_dim, hidden_dim),
            nn.GELU(),
        )
        self.return_embed = nn.Linear(1, hidden_dim)

        # Learned timestep embedding
        self.timestep_embed = nn.Embedding(config.max_episode_length, hidden_dim)

        # Token type embedding (return=0, state=1, action=2)
        self.token_type_embed = nn.Embedding(3, hidden_dim)

        # Layer norm for input
        self.embed_ln = nn.LayerNorm(hidden_dim)

        # Core Transformer (decoder-only with causal masking)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.ln_f = nn.LayerNorm(hidden_dim)

        # Action prediction head
        self.action_head = ActionHead(
            hidden_dim=hidden_dim,
            num_action_types=config.num_action_types,
            num_formations=config.num_formations,
            max_targets=config.max_threats,
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            states: (batch, seq_len, state_embed_dim)
            actions: (batch, seq_len, action_embed_dim)
            returns_to_go: (batch, seq_len, 1)
            timesteps: (batch, seq_len) - integer timestep indices
            attention_mask: (batch, seq_len) - 1 for valid, 0 for pad
        Returns:
            Action logits dict from ActionHead
        """
        batch_size, seq_len = states.shape[0], states.shape[1]
        device = states.device

        # Embed each modality
        state_embeds = self.state_embed(states)           # (B, T, H)
        action_embeds = self.action_embed(actions)        # (B, T, H)
        return_embeds = self.return_embed(returns_to_go)  # (B, T, H)

        # Add timestep embeddings
        timesteps_clamped = timesteps.clamp(0, self.config.max_episode_length - 1)
        time_embeds = self.timestep_embed(timesteps_clamped)  # (B, T, H)
        state_embeds = state_embeds + time_embeds
        action_embeds = action_embeds + time_embeds
        return_embeds = return_embeds + time_embeds

        # Interleave: [R_0, S_0, A_0, R_1, S_1, A_1, ...]
        # Stack along new dim then reshape
        stacked = torch.stack(
            [return_embeds, state_embeds, action_embeds], dim=2
        )  # (B, T, 3, H)
        sequence = stacked.reshape(batch_size, 3 * seq_len, self.config.hidden_dim)

        # Add token type embeddings
        token_types = torch.tensor(
            [0, 1, 2] * seq_len, device=device, dtype=torch.long
        )
        sequence = sequence + self.token_type_embed(token_types)

        # Layer norm
        sequence = self.embed_ln(sequence)

        # Build causal mask
        causal_mask = self._generate_causal_mask(3 * seq_len, device)

        # Build padding mask from attention_mask
        src_key_padding_mask = None
        if attention_mask is not None:
            # Expand mask from (B, T) to (B, 3T) — each timestep has 3 tokens
            expanded = attention_mask.unsqueeze(-1).expand(-1, -1, 3)
            expanded = expanded.reshape(batch_size, 3 * seq_len)
            src_key_padding_mask = ~expanded.bool()

        # Transformer forward
        hidden = self.transformer(
            sequence,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        hidden = self.ln_f(hidden)

        # Extract state token positions (index 1, 4, 7, ... = 3*t + 1)
        state_positions = torch.arange(1, 3 * seq_len, 3, device=device)
        state_hidden = hidden[:, state_positions, :]  # (B, T, H)

        # Predict actions from state positions
        return self.action_head(state_hidden)

    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper triangular) attention mask."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float("-inf"))
        return mask

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Get action for the current timestep given history.
        Convenience method for inference.
        """
        logits = self.forward(states, actions, returns_to_go, timesteps)
        return self.action_head.sample_action(
            logits, temperature=temperature, deterministic=deterministic
        )
