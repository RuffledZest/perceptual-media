import numpy as np
import pytest
import torch

from perceptual_media.core.seed import make_generator
from perceptual_media.markers.base import random_payload
from perceptual_media.metrics.decoding import (
    auc,
    ber,
    fpr_at_tpr,
    payload_recovered,
    roc,
    threshold_for_fpr,
    tpr_at_fpr,
)


def _llrs_for(payload: torch.Tensor, conf: float = 3.0) -> torch.Tensor:
    return (payload * 2 - 1) * conf  # +conf for 1, -conf for 0


def test_ber_perfect_inverted_and_batched() -> None:
    p = random_payload(3, 64, make_generator(0))
    good = _llrs_for(p)
    assert torch.equal(ber(good, p), torch.zeros(3))
    assert torch.equal(ber(-good, p), torch.ones(3))
    mixed = good.clone()
    mixed[1, :16] *= -1  # flip 16 of 64 bits in item 1 only
    assert torch.allclose(ber(mixed, p), torch.tensor([0.0, 0.25, 0.0]))


def test_ber_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        ber(torch.zeros(1, 8), torch.zeros(1, 7))


def test_payload_recovered() -> None:
    p = random_payload(2, 32, make_generator(1))
    llrs = _llrs_for(p)
    llrs[1, 5] *= -1
    assert payload_recovered(llrs, p).tolist() == [True, False]


def test_roc_perfectly_separable() -> None:
    f, t, a = roc(torch.tensor([2.0, 3.0, 4.0]), torch.tensor([-1.0, 0.0, 1.0]))
    assert a == 1.0
    assert f[0] == 0 and t[0] == 0 and f[-1] == 1 and t[-1] == 1
    assert np.all(np.diff(f) >= 0) and np.all(np.diff(t) >= 0)


def test_roc_identical_distributions_is_half() -> None:
    s = np.arange(10, dtype=float)
    assert roc(s, s)[2] == pytest.approx(0.5)
    assert auc(np.zeros(5), np.zeros(7)) == pytest.approx(0.5)  # all tied


def test_roc_matches_mann_whitney_on_random_data() -> None:
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1.0, 1, 200), rng.normal(0.0, 1, 300)
    mw = (np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    assert auc(pos, neg) == pytest.approx(mw, abs=1e-12)


def test_roc_requires_both_classes() -> None:
    with pytest.raises(ValueError):
        roc(np.array([]), np.array([1.0]))


def test_fpr_at_tpr_and_tpr_at_fpr() -> None:
    pos = np.array([0.5, 1.0, 2.0, 3.0])
    neg = np.array([0.0, 0.6, 0.1, 0.2])
    # threshold 1.0 → TPR 0.75, FPR 0; threshold 0.5 → TPR 1.0, FPR 0.25
    assert fpr_at_tpr(pos, neg, tpr=0.75) == 0.0
    assert fpr_at_tpr(pos, neg, tpr=1.0) == 0.25
    assert tpr_at_fpr(pos, neg, fpr=0.0) == 0.75
    assert tpr_at_fpr(pos, neg, fpr=0.25) == 1.0
    with pytest.raises(ValueError):
        fpr_at_tpr(pos, neg, tpr=0.0)


def test_threshold_for_fpr_respects_budget() -> None:
    rng = np.random.default_rng(3)
    neg = rng.normal(size=1000)
    for target in (0.1, 0.01, 0.001):
        t = threshold_for_fpr(neg, fpr=target)
        assert np.mean(neg >= t) <= target
        assert np.mean(neg >= t) >= target - 1 / 1000 - 1e-12  # not wastefully conservative
    assert threshold_for_fpr(neg, fpr=1.0) <= neg.min()


def test_threshold_for_fpr_warns_when_unresolvable() -> None:
    neg = np.arange(100, dtype=float)
    with pytest.warns(UserWarning, match="below 1/N"):
        t = threshold_for_fpr(neg, fpr=1e-8)
    assert t > neg.max()
    assert np.mean(neg >= t) == 0.0
