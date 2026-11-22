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
