"""Non-invasive facilitation clamp.

Inside ``u_clamp()``, facilitation u is frozen at its current profile. Resources R keep their normal
spike-driven depletion and exponential recovery. On each spike the release fraction is still computed
from the frozen value (u+ = u + U(1 - u)), so the release route sees the per-spike release it would see
if facilitation were applied but did not persist. The context manager patches ``DynamicSTFState`` and
restores the original methods on exit; no frozen module is edited.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

import model.stp as stp


@contextmanager
def u_clamp(enabled: bool = True):
    if not enabled:
        yield
        return
    cls = stp.DynamicSTFState
    orig = (cls.on_spikes, cls.half_step_recovery, cls.recover_full_step)

    def on_spikes(self, spikes, cfg, mode="dynamic", **kw):
        frozen = self.u.copy()
        out = orig[0](self, spikes, cfg, mode, **kw)
        self.u[:] = frozen
        return out

    def half_step_recovery(self, dt_ms, cfg):
        return self.u.copy(), 1.0 + (self.resources - 1.0) * np.exp(-dt_ms / (2.0 * cfg.tau_recovery_ms))

    def recover_full_step(self, dt_ms, cfg):
        self.resources[:] = 1.0 + (self.resources - 1.0) * np.exp(-dt_ms / cfg.tau_recovery_ms)

    cls.on_spikes, cls.half_step_recovery, cls.recover_full_step = on_spikes, half_step_recovery, recover_full_step
    try:
        yield
    finally:
        cls.on_spikes, cls.half_step_recovery, cls.recover_full_step = orig
