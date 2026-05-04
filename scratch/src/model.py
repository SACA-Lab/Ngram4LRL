"""Small encoder-only Transformer for sequence classification."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import SequenceClassifierOutput


class SmallTransformerClassifier(nn.Module):
    """Encoder-only Transformer with a linear classification head.

    Architecture:
      - Token embedding + learned positional embedding, layer-normed.
      - ``num_layers`` Transformer encoder blocks (pre-norm, GELU FFN).
      - Linear head on the position-zero (``<cls>``) representation.
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        max_position_embeddings: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.embedding_norm = nn.LayerNorm(d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_labels)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> SequenceClassifierOutput:
        bs, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embeddings(input_ids) + self.position_embeddings(positions)
        x = self.embedding_norm(x)
        x = self.embedding_dropout(x)

        # PyTorch encoder expects True at padding positions.
        key_padding_mask = ~attention_mask.bool() if attention_mask is not None else None

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        pooled = h[:, 0, :]
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
