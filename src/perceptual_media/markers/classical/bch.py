"""BCH error-correcting code for the fixed-size payload (brief §4.5). Default BCH(127, 64), t = 10.

Wraps ``galois`` (numba-JIT'd; the first decode call costs ~2 s to compile, then milliseconds).
The interface takes soft LLRs and hard-slices them **inside** ``decode_soft`` so a soft-input
decoder (Chase / ordered-statistics) can replace it later without touching callers.

Decoding beyond ``t`` errors: ``galois`` reports a decoding *failure* (``ok = False``) when the
syndrome cannot be resolved; a *miscorrection* to a wrong codeword is possible in principle when
an error pattern lands within ``t`` of another codeword. ``ok`` therefore means "the decoder
believes it succeeded", and the harness always compares the decoded message to the ground
truth rather than trusting ``ok`` alone.
"""

from __future__ import annotations

import numpy as np
import torch

from perceptual_media.core.types import Payload, assert_payload


class BCHCode:
    """Systematic binary BCH(n, k): message bits occupy the first ``k`` positions of a codeword."""

    def __init__(self, n: int = 127, k: int = 64) -> None:
        import galois  # heavy import (numba), deferred

        self._code = galois.BCH(n, k)
        self.n, self.k, self.t = int(self._code.n), int(self._code.k), int(self._code.t)

    def encode(self, message: Payload) -> Payload:
        """``(B, k)`` bits → ``(B, n)`` codeword bits (float32 in {0, 1}, same device)."""
        assert_payload(message)
        if message.shape[1] != self.k:
            raise ValueError(f"message must be (B, {self.k}), got {tuple(message.shape)}")
        m = message.detach().cpu().numpy().astype(np.uint8)
        c = np.asarray(self._code.encode(m), dtype=np.uint8)
        return torch.from_numpy(c).to(torch.float32).to(message.device)

    def decode_hard(self, codeword: Payload) -> tuple[Payload, torch.Tensor]:
        """``(B, n)`` received hard bits → ``(message (B, k), ok (B,) bool)``."""
        assert_payload(codeword)
        if codeword.shape[1] != self.n:
            raise ValueError(f"codeword must be (B, {self.n}), got {tuple(codeword.shape)}")
        r = codeword.detach().cpu().numpy().astype(np.uint8)
        msg, n_err = self._code.decode(r, errors=True)
        msg = np.asarray(msg, dtype=np.uint8)
        n_err = np.atleast_1d(np.asarray(n_err))
        ok = torch.from_numpy(n_err >= 0)
        return torch.from_numpy(msg).to(torch.float32).to(codeword.device), ok.to(codeword.device)

    def decode_soft(self, llrs: torch.Tensor) -> tuple[Payload, torch.Tensor]:
        """``(B, n)`` LLRs (``> 0`` ⇒ bit 1) → ``(message, ok)``. Hard-decision for now."""
        return self.decode_hard((llrs > 0).to(torch.float32))
