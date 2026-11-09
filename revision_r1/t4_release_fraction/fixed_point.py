"""Release-fraction fixed point of the oscillation frequency (R1-T4)."""
from __future__ import annotations

import numpy as np

from revision_r1.t2_efficacy_stats.metrics import u_plus_periodic


def make_F(table, p_main: float):
    """F(rho, p) for one tau_r row.

    table: rows with keys rho, p_target, freq_hz (valid final cells of that row).  The main curve (p = p_main)
    must cover the rho range; sensitivity points at other p give a per-rho linear correction in p, interpolated
    in rho and applied piecewise-linearly between the p values that were run.
    """
    import pandas as pd
    t = pd.DataFrame(table)
    main = t[np.isclose(t.p_target, p_main)].sort_values("rho")
    rho_m, f_m = main.rho.to_numpy(float), main.freq_hz.to_numpy(float)
    sens = t[~np.isclose(t.p_target, p_main)]
    p_levels = sorted(set(np.round(sens.p_target, 6))) if len(sens) else []

    def F_main(rho):
        return float(np.interp(rho, rho_m, f_m, left=np.nan, right=np.nan))

    def delta_at(p_level, rho):
        s = sens[np.isclose(sens.p_target, p_level)].sort_values("rho")
        d = s.freq_hz.to_numpy(float) - np.array([F_main(r) for r in s.rho])
        return float(np.interp(rho, s.rho.to_numpy(float), d))       # constant extrapolation in rho

    def F(rho, p=p_main):
        base = F_main(rho)
        if not np.isfinite(base) or not p_levels or np.isclose(p, p_main):
            return base
        pts = sorted([(p_main, 0.0)] + [(pl, delta_at(pl, rho)) for pl in p_levels])
        ps, ds = zip(*pts)
        if p < ps[0] - 1e-9 or p > ps[-1] + 1e-9:
            return float("nan")                                       # no extrapolation in p
        return base + float(np.interp(p, ps, ds))
    return F


def solve_fixed_point(F, U: float, tau_f: float, p: float, spikes_per_cycle: float = 1.0,
                      f_lo: float = 1.0, f_hi: float = 40.0, step: float = 0.005):
    """Unique root of g(f) = F(u+(f), p) - f (u+ rises with f, F falls with rho): returns (f*, u+(f*))."""
    fs = np.arange(f_lo, f_hi + step, step)
    g = np.array([F(u_plus_periodic(f, U, tau_f, spikes_per_cycle), p) - f for f in fs])
    ok = np.isfinite(g)
    if not ok.any():
        return float("nan"), float("nan")
    idx = np.flatnonzero(ok[:-1] & ok[1:] & (np.sign(g[:-1]) != np.sign(g[1:])))
    if idx.size == 0:
        return float("nan"), float("nan")
    i = idx[0]
    f_star = float(fs[i] - g[i] * (fs[i + 1] - fs[i]) / (g[i + 1] - g[i]))
    return f_star, u_plus_periodic(f_star, U, tau_f, spikes_per_cycle)
