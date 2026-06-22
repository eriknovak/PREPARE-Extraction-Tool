import torch

from app.training.gliner_trainer import _per_element_mean_loss


def test_normalized_loss_equals_mean_reduction():
    """The reported loss must equal what ``loss_reduction='mean'`` would yield.

    GLiNER's Trainer backprops a sum-reduced loss (sum over every score element);
    the model's own ``mean`` reduction is exactly ``sum / numel``. Our reporting
    normalization must reproduce that, so the logged curve matches the mean loss
    without altering the (summed) gradient signal.
    """
    # Per-element losses shaped like GLiNER's score tensor
    # (batch * seq_len * num_classes) at a realistic size.
    all_losses = torch.full((8, 384, 12), 0.5)

    summed = all_losses.sum()
    numel = all_losses.numel()

    reported = _per_element_mean_loss(summed, numel)

    # Matches mean reduction exactly.
    assert torch.allclose(reported, all_losses.mean())
    # The raw sum is absurd (~18k here, thousands-to-hundreds-of-thousands in
    # practice); the reported value lands in a sane single/low-double-digit range.
    assert summed.item() > 1000.0
    assert reported.item() < 100.0


def test_normalized_loss_accepts_plain_floats():
    # Works for python scalars too (e.g. an already-detached float).
    assert _per_element_mean_loss(400_000.0, 50_000) == 8.0


def test_zero_numel_is_safe():
    # Degenerate/empty batches must not divide by zero; return the input as-is.
    assert _per_element_mean_loss(123.0, 0) == 123.0
