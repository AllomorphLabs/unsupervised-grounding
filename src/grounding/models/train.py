"""Training one model of a pair on its own stream.

    uv run python -m grounding.models.train runs/pilot/pair-000 --stream lang

Reads the pair's public data only, trains for ``steps`` steps or until the time
budget runs out, and writes to ``<pair>/models/<stream>/``:

    model.pt        weights of the best model, and the configs needed to rebuild it
    metrics.jsonl   training and validation loss over time
    summary.json    the best step, its validation loss, and the final held-out loss
    run.json        configs, training seed, git commit, package versions

A share of the training episodes is set aside for validation, and the model
kept is the one with the lowest validation loss. The held-out episodes are
reserved for scoring: they play no part in training or in choosing the model,
and their loss is reported once, at the end, for information only.

The training seed comes from the pair's root seed, from a part of the stream's
seed reserved for training, so every pair has an independent training seed.
"""

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from grounding.data.corpus import PARTS
from grounding.data.load import load_lang, load_obs
from grounding.data.tokens import STOP
from grounding.models.transformer import FrameModel, LanguageModel, ModelConfig, count_parameters
from grounding.runs.record import make_record, read_record, write_record
from grounding.runs.seeds import part_rngs


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20_000
    batch_size: int = 32
    lr: float = 3e-4
    warmup: int = 200
    weight_decay: float = 0.1
    eval_every: int = 500
    max_minutes: float = 10.0
    validation: float = 0.1  # share of training episodes used to choose the best step
    amp: bool = False  # bfloat16 mixed precision on the GPU: faster, slightly less exact


@dataclass(frozen=True)
class ModelsConfig:
    """Model and training settings for both streams, read from ``configs/*-models.yaml``."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> ModelsConfig:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(ModelConfig(**raw.get("model", {})), TrainConfig(**raw.get("train", {})))


def training_seed(root: int, stream: str) -> int:
    return int(part_rngs(root, stream, PARTS[stream])["train"].integers(2**31))


def _episodes(starts: np.ndarray, total: int) -> list[tuple[int, int]]:
    ends = np.append(starts[1:], total)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


class Batches:
    """Padded batches of whole episodes from one stream."""

    def __init__(
        self, stream: str, data_dir: Path, device: torch.device, validation: float, seed: int
    ):
        self.stream, self.device = stream, device
        if stream == "lang":
            s = load_lang(data_dir)
            self.data = torch.from_numpy(s.tokens.astype(np.int64))
            self.vocab = len(s.words)
            self.anchor = s.words.index(STOP)  # sentence ends: public structure
        elif stream == "obs":
            s = load_obs(data_dir)
            self.data = torch.from_numpy(s.frames.astype(np.float32))
            self.frame_size = s.frames.shape[1]
        else:
            raise ValueError(f"unknown stream {stream!r}")
        spans = _episodes(s.episode_start, len(self.data))
        train = [sp for sp, h in zip(spans, s.held_out, strict=True) if not h]
        self.held_out = [sp for sp, h in zip(spans, s.held_out, strict=True) if h]
        # The validation split has its own random stream, separate from batch sampling.
        order = np.random.default_rng([seed, 1]).permutation(len(train))
        n_val = round(validation * len(train))
        self.validation = [train[i] for i in sorted(order[:n_val])]
        self.train = [train[i] for i in sorted(order[n_val:])]
        self.longest = max(b - a for a, b in spans)

    def make_model(self, cfg: ModelConfig) -> torch.nn.Module:
        if self.stream == "lang":
            return LanguageModel(self.vocab, cfg, anchor=self.anchor if cfg.future else None)
        return FrameModel(self.frame_size, cfg)

    def batch(self, spans: list[tuple[int, int]]):
        t = max(b - a for a, b in spans)
        if self.stream == "lang":
            out = torch.full((len(spans), t), self.vocab, dtype=torch.int64)
            for i, (a, b) in enumerate(spans):
                out[i, : b - a] = self.data[a:b]
            return (out.to(self.device),)
        out = torch.zeros((len(spans), t, self.frame_size))
        mask = torch.zeros((len(spans), t), dtype=torch.bool)
        for i, (a, b) in enumerate(spans):
            out[i, : b - a] = self.data[a:b]
            mask[i, : b - a] = True
        return out.to(self.device), mask.to(self.device)


def _lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def mean_loss(model, data: Batches, spans_all: list[tuple[int, int]], batch_size: int) -> float:
    """Mean loss over ``spans_all``, batch by batch, weighted by batch size."""
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(spans_all), batch_size):
        spans = spans_all[i : i + batch_size]
        total += model.loss(*data.batch(spans)).item() * len(spans)
        n += len(spans)
    model.train()
    return total / n if n else float("nan")


def train(
    pair: Path, stream: str, cfg: ModelsConfig, device: str | None = None, out: Path | None = None
) -> dict[str, Any]:
    """Train one model and write it under ``<pair>/models/<stream>/``. Returns a summary."""
    pair = Path(pair)
    out = Path(out) if out else pair / "models" / stream
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; refusing to overwrite a trained model")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    root = read_record(pair / "data").seed
    seed = training_seed(root, stream)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    data = Batches(stream, pair / "data", dev, cfg.train.validation, seed)
    model_cfg = ModelConfig(**{**asdict(cfg.model), "context": data.longest})
    model = data.make_model(model_cfg).to(dev)
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    tc = cfg.train
    started = time.monotonic()
    metrics_file = (out / "metrics.jsonl").open("w", encoding="utf-8")
    stopped_by_time = False
    step = 0
    best = {"step": -1, "validation_loss": math.inf, "state": None}
    for step in range(tc.steps):
        if (time.monotonic() - started) / 60 > tc.max_minutes:
            stopped_by_time = True
            break
        for g in opt.param_groups:
            g["lr"] = _lr(step, tc)
        picks = rng.choice(len(data.train), size=min(tc.batch_size, len(data.train)), replace=False)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=tc.amp and dev.type == "cuda"):
            loss = model.loss(*data.batch([data.train[i] for i in picks]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % tc.eval_every == 0 or step == tc.steps - 1:
            val = mean_loss(model, data, data.validation, tc.batch_size)
            if val < best["validation_loss"]:
                state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best = {"step": step, "validation_loss": val, "state": state}
            row = {
                "step": step,
                "train_loss": loss.item(),
                "validation_loss": val,
                "minutes": (time.monotonic() - started) / 60,
            }
            metrics_file.write(json.dumps(row) + "\n")
            metrics_file.flush()
    else:
        step = tc.steps
    metrics_file.close()
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    summary = {
        "stream": stream,
        "steps": step,
        "stopped_by_time": stopped_by_time,
        "best_step": best["step"],
        "validation_loss": best["validation_loss"],
        "held_out_loss": mean_loss(model, data, data.held_out, tc.batch_size),
        "parameters": count_parameters(model),
        "minutes": (time.monotonic() - started) / 60,
        "device": str(dev),
    }
    torch.save(
        {
            "stream": stream,
            "model": asdict(model_cfg),
            "size": data.vocab if stream == "lang" else data.frame_size,
            "anchor": data.anchor if stream == "lang" and model_cfg.future else None,
            "state": model.state_dict(),
            "summary": summary,
        },
        out / "model.pt",
    )
    config = {"model": asdict(model_cfg), "train": asdict(tc), "stream": stream, "pair_seed": root}
    write_record(out, make_record(config, seed))
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# Decision 15: the pilot language model is larger than the frame model (decision 13).
DEFAULT_CONFIGS = {
    "lang": Path("configs/pilot-lang.yaml"),
    "obs": Path("configs/pilot-models.yaml"),
}


def load_model(model_dir: Path, device: str = "cpu") -> torch.nn.Module:
    saved = torch.load(Path(model_dir) / "model.pt", map_location=device, weights_only=True)
    cfg = ModelConfig(**saved["model"])
    if saved["stream"] == "lang":
        model = LanguageModel(saved["size"], cfg, anchor=saved.get("anchor"))
    else:
        model = FrameModel(saved["size"], cfg)
    model.load_state_dict(saved["state"])
    return model.to(device).eval()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("pair", type=Path)
    p.add_argument("--stream", choices=("lang", "obs"), required=True)
    p.add_argument("--config", type=Path, default=None, help="default: DEFAULT_CONFIGS[stream]")
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None, help="model folder; default: models/<stream>")
    args = p.parse_args()
    config = ModelsConfig.from_yaml(args.config or DEFAULT_CONFIGS[args.stream])
    summary = train(args.pair, args.stream, config, args.device, out=args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
