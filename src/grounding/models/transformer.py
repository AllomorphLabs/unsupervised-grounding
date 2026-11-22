"""The two models, built on one small causal transformer.

Both models read one episode at a time and predict what comes next, using only
what came before (causal attention):

- ``LanguageModel`` reads token numbers and predicts the next token.
- ``FrameModel`` reads state-vector frames and predicts each value of the next
  frame as an independent yes/no. It is not told which values form a block,
  so it must learn the structure of the world itself (technical decision 13).

Episodes in a batch are padded at the end. Causal attention means real
positions never attend to padding, so no padding mask is needed; padded
positions are simply left out of the loss.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    layers: int = 4
    width: int = 128
    heads: int = 4
    context: int = 1024  # longest episode, in tokens or frames
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.heads = cfg.heads
        self.dropout = cfg.dropout
        self.norm1 = nn.LayerNorm(cfg.width)
        self.qkv = nn.Linear(cfg.width, 3 * cfg.width)
        self.proj = nn.Linear(cfg.width, cfg.width)
        self.norm2 = nn.LayerNorm(cfg.width)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.width, 4 * cfg.width), nn.GELU(), nn.Linear(4 * cfg.width, cfg.width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, w = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(w, dim=-1)
        q, k, v = (z.view(b, t, self.heads, w // self.heads).transpose(1, 2) for z in (q, k, v))
        drop = self.dropout if self.training else 0.0
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, w))
        return x + self.mlp(self.norm2(x))


class Backbone(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.width % cfg.heads:
            raise ValueError("width must be divisible by heads")
        self.cfg = cfg
        self.pos = nn.Embedding(cfg.context, cfg.width)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self, x: torch.Tensor, all_layers: bool = False):
        """Hidden states: the last layer's, or every layer's if ``all_layers``."""
        t = x.shape[1]
        if t > self.cfg.context:
            raise ValueError(f"episode of length {t} exceeds context {self.cfg.context}")
        x = x + self.pos(torch.arange(t, device=x.device))
        layers = [x]
        for block in self.blocks:
            x = block(x)
            layers.append(x)
        x = self.norm(x)
        return torch.stack(layers[:-1] + [x]) if all_layers else x


class LanguageModel(nn.Module):
    """Next-token prediction. Token number ``vocab`` is reserved for padding."""

    def __init__(self, vocab: int, cfg: ModelConfig):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab + 1, cfg.width)
        self.backbone = Backbone(cfg)
        self.head = nn.Linear(cfg.width, vocab)

    @property
    def pad(self) -> int:
        return self.vocab

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.embed(tokens)))

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Every layer's hidden states, shape (layers + 1, batch, time, width)."""
        return self.backbone(self.embed(tokens), all_layers=True)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self(tokens[:, :-1])
        return F.cross_entropy(
            logits.reshape(-1, self.vocab), tokens[:, 1:].reshape(-1), ignore_index=self.pad
        )


class FrameModel(nn.Module):
    """Next-frame prediction, each value an independent yes/no."""

    def __init__(self, frame_size: int, cfg: ModelConfig):
        super().__init__()
        self.frame_size = frame_size
        self.embed = nn.Linear(frame_size, cfg.width)
        self.backbone = Backbone(cfg)
        self.head = nn.Linear(cfg.width, frame_size)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.embed(frames)))

    def hidden(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.embed(frames), all_layers=True)

    def loss(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask[b, t]`` is true where frame ``t`` of episode ``b`` is real, not padding."""
        logits = self(frames[:, :-1])
        target, keep = frames[:, 1:], mask[:, 1:]
        per_value = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return per_value[keep].mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
