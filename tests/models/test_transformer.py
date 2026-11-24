import torch

from grounding.models.transformer import FrameModel, LanguageModel, ModelConfig, count_parameters

TINY = ModelConfig(layers=2, width=16, heads=2, context=32)


def test_pilot_size_is_about_one_million_parameters():
    n = count_parameters(LanguageModel(24, ModelConfig(context=1024)))
    assert 0.8e6 < n < 1.2e6


def test_language_model_is_causal():
    torch.manual_seed(0)
    model = LanguageModel(10, TINY).eval()
    a = torch.randint(0, 10, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 10  # change only the future
    assert torch.allclose(model(a)[0, :8], model(b)[0, :8], atol=1e-6)
    assert not torch.allclose(model(a)[0, 8:], model(b)[0, 8:])


def test_padding_does_not_change_real_positions():
    torch.manual_seed(0)
    model = LanguageModel(10, TINY).eval()
    x = torch.randint(0, 10, (1, 7))
    padded = torch.cat([x, torch.full((1, 5), model.pad)], dim=1)
    assert torch.allclose(model(x)[0], model(padded)[0, :7], atol=1e-6)


def test_frame_model_is_causal_and_masks_padding():
    torch.manual_seed(0)
    model = FrameModel(9, TINY).eval()
    frames = torch.randint(0, 2, (2, 10, 9)).float()
    changed = frames.clone()
    changed[:, 6:] = 1 - changed[:, 6:]
    assert torch.allclose(model(frames)[:, :6], model(changed)[:, :6], atol=1e-6)
    # Garbage in padded frames must not move the loss.
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[1, 5:] = False
    noisy = frames.clone()
    noisy[1, 5:] = torch.rand(5, 9)
    assert torch.isclose(model.loss(frames, mask), model.loss(noisy, mask))


def test_hidden_states_cover_every_layer():
    model = LanguageModel(10, TINY)
    h = model.hidden(torch.randint(0, 10, (3, 5)))
    assert h.shape == (TINY.layers + 1, 3, 5, TINY.width)


def test_predictive_state_head_needs_the_sentence_end_token():
    import pytest

    with pytest.raises(ValueError):
        LanguageModel(10, ModelConfig(layers=1, width=16, heads=2, context=32, future=4))


def test_predictive_state_head_trains_the_summary():
    torch.manual_seed(0)
    cfg = ModelConfig(layers=1, width=16, heads=2, context=32, future=4, summary=8)
    model = LanguageModel(10, cfg, anchor=9)
    x = torch.randint(0, 9, (2, 20))
    x[:, ::5] = 9  # sentence ends
    plain = LanguageModel(10, ModelConfig(layers=1, width=16, heads=2, context=32))
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if k in plain.state_dict()})
    model.eval()
    # The head adds the future loss on top of the next-token loss.
    assert model.loss(x).item() > plain.eval().loss(x).item()
    assert model.summaries(x).shape == (2, 20, 8)
    model.train()
    model.loss(x).backward()
    assert model.down.weight.grad.abs().sum() > 0


def test_predictive_state_validation_loss_is_deterministic():
    torch.manual_seed(0)
    cfg = ModelConfig(layers=1, width=16, heads=2, context=64, future=4, summary=8)
    model = LanguageModel(10, cfg, anchor=9).eval()
    x = torch.randint(0, 9, (1, 60))
    x[:, ::3] = 9  # more sentence ends than are sampled per episode
    assert model.loss(x).item() == model.loss(x).item()


JOINT = ModelConfig(layers=2, width=16, heads=2, context=32, joint=True, joint_hidden=32)


def test_joint_head_sees_only_earlier_values_of_the_next_frame():
    torch.manual_seed(0)
    model = FrameModel(6, JOINT).eval()
    h = torch.randn(3, JOINT.width)
    x = torch.randint(0, 2, (3, 6)).float()
    y = x.clone()
    y[:, 3:] = 1 - y[:, 3:]  # change values 3 onwards
    a, b = model.next_logits(h, x), model.next_logits(h, y)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-6)  # value 3's logit can't see value 3
    assert not torch.allclose(a[:, 4:], b[:, 4:])


def test_joint_head_gives_a_distribution_over_whole_frames():
    torch.manual_seed(0)
    model = FrameModel(4, JOINT).eval()
    h = torch.randn(1, JOINT.width).expand(16, -1)
    every = torch.tensor([[(i >> k) & 1 for k in range(4)] for i in range(16)]).float()
    with torch.no_grad():
        total = model.log_prob_next(h, every).exp().sum()
    assert torch.isclose(total, torch.tensor(1.0), atol=1e-5)


def test_joint_frame_model_is_causal_and_samples_frames():
    torch.manual_seed(0)
    model = FrameModel(9, JOINT).eval()
    frames = torch.randint(0, 2, (2, 10, 9)).float()
    changed = frames.clone()
    changed[:, 6:] = 1 - changed[:, 6:]
    assert torch.allclose(model.context(frames)[:, :6], model.context(changed)[:, :6], atol=1e-6)
    mask = torch.ones(2, 10, dtype=torch.bool)
    assert model.loss(frames, mask).item() > 0
    s = model.sample_next(model.context(frames)[:, -1])
    assert s.shape == (2, 9) and set(s.unique().tolist()) <= {0.0, 1.0}
