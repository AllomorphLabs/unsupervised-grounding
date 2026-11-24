"""The two models, built on one small causal transformer.

Both models read one episode at a time and predict what comes next, using only
what came before (causal attention):

- ``LanguageModel`` reads token numbers and predicts the next token. With
  ``future > 0`` it also has a predictive-state head (technical decision 17): at
  each sentence end, a ``summary``-sized vector must predict the next ``future``
  tokens on its own, through a small decoder. What comes next depends on where
  everything is, so the summary is pushed to hold the whole state of the world.
- ``FrameModel`` reads state-vector frames and predicts the next frame. By
  default each value is an independent yes/no. With ``joint`` (technical
  decision 18) the values are predicted one after another, each given the
  history and the values before it, so the model gives the probability of a
  whole frame and can sample valid frames. Either way it is not told which
  values form a block, so it must learn the structure of the world itself
  (technical decision 13).

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
    future: int = 0  # language only: tokens the summary must predict; 0 turns the head off
    summary: int = 64  # language only: size of the summary at each sentence end
    future_layers: int = 2  # language only: layers in the decoder that reads the summary
    joint: bool = False  # frames only: predict whole next frames, value by value
    joint_hidden: int = 256  # frames only: hidden units in the joint head


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
    """Next-token prediction. Token number ``vocab`` is reserved for padding.

    ``anchor`` is the sentence-end token, where summaries are taken; it is public
    structure. It is needed only when ``cfg.future > 0``.
    """

    def __init__(self, vocab: int, cfg: ModelConfig, anchor: int | None = None):
        super().__init__()
        self.vocab = vocab
        self.anchor = anchor
        self.future = cfg.future
        self.embed = nn.Embedding(vocab + 1, cfg.width)
        self.backbone = Backbone(cfg)
        self.head = nn.Linear(cfg.width, vocab)
        if cfg.future:
            if anchor is None:
                raise ValueError("a predictive-state head needs the sentence-end token")
            dec = ModelConfig(
                layers=cfg.future_layers, width=cfg.width, heads=cfg.heads, context=cfg.future + 1
            )
            self.down = nn.Linear(cfg.width, cfg.summary)
            self.up = nn.Linear(cfg.summary, cfg.width)
            self.future_embed = nn.Embedding(vocab + 1, cfg.width)
            self.future_decoder = Backbone(dec)
            self.future_head = nn.Linear(cfg.width, vocab)

    @property
    def pad(self) -> int:
        return self.vocab

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(self.embed(tokens)))

    def hidden(self, tokens: torch.Tensor) -> torch.Tensor:
        """Every layer's hidden states, shape (layers + 1, batch, time, width)."""
        return self.backbone(self.embed(tokens), all_layers=True)

    def summaries(self, tokens: torch.Tensor) -> torch.Tensor:
        """The predictive-state summary at every position, shape (batch, time, summary)."""
        return self.down(self.backbone(self.embed(tokens)))

    def _anchors(self, tokens: torch.Tensor, per_episode: int = 16):
        """Sentence ends to train the summary at: random ones while training, evenly
        spaced ones otherwise, so validation loss is deterministic."""
        rows, cols = [], []
        for b in range(tokens.shape[0]):
            ends = torch.nonzero(tokens[b] == self.anchor).flatten()
            if len(ends) == 0:
                continue
            if len(ends) > per_episode:
                if self.training:
                    pick = torch.randperm(len(ends), device=ends.device)[:per_episode]
                else:
                    pick = torch.linspace(0, len(ends) - 1, per_episode, device=ends.device).long()
                ends = ends[pick]
            rows.append(torch.full_like(ends, b))
            cols.append(ends)
        return torch.cat(rows), torch.cat(cols)

    def loss(self, tokens: torch.Tensor) -> torch.Tensor:
        h = self.backbone(self.embed(tokens))
        logits = self.head(h[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab), tokens[:, 1:].reshape(-1), ignore_index=self.pad
        )
        if not self.future:
            return loss
        rows, cols = self._anchors(tokens)
        # The next `future` tokens after each anchor, padded past the end of the episode.
        padded = F.pad(tokens, (0, self.future), value=self.pad)
        fut = padded[rows[:, None], cols[:, None] + 1 + torch.arange(self.future, device=h.device)]
        x = torch.cat(
            [self.up(self.down(h[rows, cols]))[:, None], self.future_embed(fut[:, :-1])], 1
        )
        fut_logits = self.future_head(self.future_decoder(x))
        fut_loss = F.cross_entropy(
            fut_logits.reshape(-1, self.vocab), fut.reshape(-1), ignore_index=self.pad
        )
        return loss + fut_loss


class MaskedLinear(nn.Linear):
    def __init__(self, n_in: int, n_out: int, mask: torch.Tensor, bias: bool = True):
        super().__init__(n_in, n_out, bias=bias)
        self.register_buffer("mask", mask.float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class JointHead(nn.Module):
    """Logits for each value of the next frame, given a context vector and the
    next frame's earlier values (a masked network, after MADE). Value ``v``'s
    logit depends only on values ``0..v-1``, so the product of the yes/no
    probabilities is the probability of the whole frame."""

    def __init__(self, frame_size: int, width: int, hidden: int):
        super().__init__()
        d = frame_size
        m1 = torch.arange(hidden) % d  # how many earlier values each hidden unit may see
        m2 = torch.arange(hidden) % d
        self.l1 = MaskedLinear(d, hidden, m1[:, None] >= torch.arange(1, d + 1)[None])
        self.l2 = MaskedLinear(hidden, hidden, m2[:, None] >= m1[None])
        self.out = MaskedLinear(hidden, d, torch.arange(d)[:, None] >= m2[None])
        self.c1, self.c2, self.co = (nn.Linear(width, n) for n in (hidden, hidden, d))

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        a = F.gelu(self.l1(x) + self.c1(h))
        a = F.gelu(self.l2(a) + self.c2(h))
        return self.out(a) + self.co(h)


class FrameModel(nn.Module):
    """Next-frame prediction: each value an independent yes/no, or, with
    ``cfg.joint``, whole frames value by value."""

    def __init__(self, frame_size: int, cfg: ModelConfig):
        super().__init__()
        self.frame_size = frame_size
        self.embed = nn.Linear(frame_size, cfg.width)
        self.backbone = Backbone(cfg)
        self.joint = JointHead(frame_size, cfg.width, cfg.joint_hidden) if cfg.joint else None
        if self.joint is None:
            self.head = nn.Linear(cfg.width, frame_size)

    def context(self, frames: torch.Tensor) -> torch.Tensor:
        """Last-layer hidden state at every frame: what the model knows so far."""
        return self.backbone(self.embed(frames))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Independent yes/no logits for the frame after each position."""
        if self.joint is not None:
            raise TypeError("a joint frame model needs the next frame: use next_logits")
        return self.head(self.context(frames))

    def next_logits(self, h: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """Logits for ``nxt`` (a next frame for each context ``h``). With the joint head,
        value ``v``'s logit is conditioned on ``nxt``'s values before ``v``."""
        return self.joint(h, nxt) if self.joint is not None else self.head(h)

    def log_prob_next(self, h: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """Log probability of each whole next frame."""
        logits = self.next_logits(h, nxt)
        return -F.binary_cross_entropy_with_logits(logits, nxt, reduction="none").sum(-1)

    @torch.no_grad()
    def sample_next(
        self, h: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """One next frame per context ``h`` (shape ``(n, width)``)."""
        n = h.shape[0]
        x = torch.zeros(n, self.frame_size, device=h.device)
        if self.joint is None:
            p = torch.sigmoid(self.head(h))
            return (torch.rand(p.shape, device=h.device, generator=generator) < p).float()
        j = self.joint
        c1, c2, co = j.c1(h), j.c2(h), j.co(h)  # the context's part, the same for every value
        for v in range(self.frame_size):
            a = F.gelu(j.l1(x) + c1)
            a = F.gelu(j.l2(a) + c2)
            logit = a @ (j.out.weight[v] * j.out.mask[v]) + j.out.bias[v] + co[:, v]
            x[:, v] = (
                torch.rand(n, device=h.device, generator=generator) < torch.sigmoid(logit)
            ).float()
        return x

    def hidden(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.embed(frames), all_layers=True)

    def loss(self, frames: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """``mask[b, t]`` is true where frame ``t`` of episode ``b`` is real, not padding."""
        target, keep = frames[:, 1:], mask[:, 1:]
        logits = self.next_logits(self.context(frames[:, :-1]), target)
        per_value = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        return per_value[keep].mean()


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
