"""Detection-theory metrics (brief §4.2): BER, payload recovery, ROC / AUC, FPR helpers.

Pure torch/NumPy. Inputs follow ``markers.base.DecodeResult``: ``llrs`` are ``(B, n_bits)``
with ``llr > 0`` meaning bit 1; ``scores`` are ``(N,)`` presence statistics, higher = more
likely marked.

On false positives (brief §4.2): a detector scanning at 30 fps runs ~2.6 M trials per day.
A per-frame FPR of 1e-3 is a false positive every 33 s. Anything meant for continuous scanning
needs ~1e-8 per frame or a multi-frame confirmation protocol; ``threshold_for_fpr`` warns when
asked for a target that the control set is too small to estimate.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch

from perceptual_media.core.types import Payload, assert_payload


def ber(llrs: torch.Tensor, payload: Payload) -> torch.Tensor:
    """``(B,)`` bit-error rate: fraction of hard-sliced bits (``llr > 0``) that differ from ``payload``."""
    assert_payload(payload)
    if llrs.shape != payload.shape:
        raise ValueError(f"llrs {tuple(llrs.shape)} vs payload {tuple(payload.shape)}")
    hard = (llrs > 0).to(torch.float32)
    return (hard != payload).to(torch.float32).mean(dim=1)


def payload_recovered(llrs: torch.Tensor, payload: Payload) -> torch.Tensor:
    """``(B,)`` bool: every bit correct after hard slicing (no ECC; the ECC-aware path arrives in Task 16)."""
    return ber(llrs, payload) == 0


def roc(scores_marked: torch.Tensor | np.ndarray, scores_unmarked: torch.Tensor | np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """ROC of a threshold detector ``score >= t``.

    Returns ``(fpr, tpr, auc)`` with ``fpr``/``tpr`` monotone arrays of length ``N + 1`` starting
    at (0, 0) and ending at (1, 1). Ties are handled so that AUC equals the Mann–Whitney
    statistic: identical distributions give AUC = 0.5, perfect separation gives 1.0.
    """
    pos = np.asarray(_np(scores_marked), dtype=np.float64).ravel()
    neg = np.asarray(_np(scores_unmarked), dtype=np.float64).ravel()
    if pos.size == 0 or neg.size == 0:
        raise ValueError("need at least one marked and one unmarked score")
    scores = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(pos.size, bool), np.zeros(neg.size, bool)])
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    # collapse ties: keep the last index of each distinct score
    distinct = np.r_[scores[1:] != scores[:-1], True]
    tp, fp = tp[distinct], fp[distinct]
    tpr = np.r_[0.0, tp / pos.size]
    fpr = np.r_[0.0, fp / neg.size]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def auc(scores_marked: torch.Tensor | np.ndarray, scores_unmarked: torch.Tensor | np.ndarray) -> float:
    """Area under the ROC curve (see ``roc``)."""
    return roc(scores_marked, scores_unmarked)[2]


def fpr_at_tpr(scores_marked: torch.Tensor | np.ndarray, scores_unmarked: torch.Tensor | np.ndarray, tpr: float = 0.95) -> float:
    """Smallest FPR achievable while keeping TPR ≥ ``tpr``."""
    if not 0 < tpr <= 1:
        raise ValueError("tpr must be in (0, 1]")
    f, t, _ = roc(scores_marked, scores_unmarked)
    ok = np.nonzero(t >= tpr)[0]
    return float(f[ok[0]])


def tpr_at_fpr(scores_marked: torch.Tensor | np.ndarray, scores_unmarked: torch.Tensor | np.ndarray, fpr: float = 1e-3) -> float:
    """Largest TPR achievable with FPR ≤ ``fpr``."""
    if not 0 <= fpr <= 1:
        raise ValueError("fpr must be in [0, 1]")
    f, t, _ = roc(scores_marked, scores_unmarked)
    ok = np.nonzero(f <= fpr)[0]
    return float(t[ok[-1]])


def threshold_for_fpr(scores_unmarked: torch.Tensor | np.ndarray, fpr: float = 1e-3) -> float:
    """Detection threshold ``t`` such that ``P(score_unmarked >= t) <= fpr`` on the control set.

    Empirical: with ``N`` control scores the smallest resolvable FPR is ``1/N``; asking for
    less warns and returns a threshold just above the maximum control score.
    """
    neg = np.sort(np.asarray(_np(scores_unmarked), dtype=np.float64).ravel())
    n = neg.size
    if n == 0:
        raise ValueError("need at least one unmarked score")
    if not 0 <= fpr <= 1:
        raise ValueError("fpr must be in [0, 1]")
    if fpr < 1 / n:
        warnings.warn(
            f"target FPR {fpr:g} is below 1/N = {1 / n:g} for N={n} control scores; "
            f"returning a threshold above the max control score. Collect more unmarked "
            f"samples or use a multi-frame protocol (brief §4.2).",
            stacklevel=2,
        )
        return float(np.nextafter(neg[-1], np.inf))
    k = int(np.floor(fpr * n))  # allow at most k control scores at/above threshold
    if k == 0:
        return float(np.nextafter(neg[-1], np.inf))
    return float(np.nextafter(neg[n - k - 1], np.inf)) if n - k - 1 >= 0 else float(neg[0])


def _np(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
