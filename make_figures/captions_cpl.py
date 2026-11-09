# -*- coding: utf-8 -*-
"""Main-figure legends for the Chinese Physics Letters manuscript (Figs. 1-4)."""

CAPTIONS = {
    'Fig1': (
        'Fig. 1. Persistent activity and theta-band inhibitory oscillations in the full model '
        '(reference point; one trial with σ_η = 90 pA). (a) Network schematic. (b) Excitatory spikes '
        '(black bar, cue); right, mean firing rate of each E neuron over 1–8 s; gray bar, interval '
        'expanded in (c), (e), and (f). (c) Excitatory (top) and inhibitory (bottom) spikes. (d) Power '
        'spectral density of I_inh over 1–8 s, with dominant frequency f_0 = 5.08 Hz and harmonics at '
        '2f_0 and 3f_0; bar, 4–12 Hz. (e) Facilitation u_i and (f) depletion D_i, reconstructed from '
        'the recorded spikes; color scales start at U and 0 (square-root color scale for D_i).'
    ),
    'Fig2': (
        'Fig. 2. Oscillation frequency in route-isolated models and its dependence on release fraction. '
        '(a) Schematic of the output and release routes through which facilitation u affects E→I '
        'transmission. (b) Mean-subtracted inhibitory currents from example trials at the reference '
        'point (mean-efficacy matching; common scale bar, 0.5 nA). (c) Dominant frequencies of the '
        'output-only and release-only models against the full-model frequency at 42 (τ_f, τ_r) settings '
        '(means of four trials); dotted line, identity; black rings, reference point. (d) STP-modulated '
        'NMDA gating contribution c(t) = u^+ s^NMDA(t)[1 − ρ exp(−t/τ_r)] of one E neuron firing at the '
        'reference frequency, normalized to its peak without depletion, for ρ = U and ρ = ū_rel. '
        '(e) F(ρ, τ_r, p̄) for τ_r = 5, 20, and 80 ms (four trials per point); triangle and large '
        'circle, output-only and full models at the reference point; side bar, 4–12 Hz. (f) Solutions '
        'of f = F[u^+(f), τ_r, p̄] against the full-model frequency for the (τ_f, τ_r) sweep (filled '
        'markers; symbols as in (e)) and the U sweep (open squares); dotted line, identity.'
    ),
    'Fig3': (
        'Fig. 3. Restoration after activity-only and selective STP-state shifts in the full model. '
        '(a)–(c) Example confirmation trial [black rings in (d)] following shifts of activity alone (a), '
        'activity and {u_i} (b), or activity and {R_i} (c). Top: extents above half maximum of the '
        'folded s^NMDA (blue), u_i − U (orange), and D_i (green) profiles 10 ms after the shift. '
        'Bottom: φ_s and ψ_u relative to the old position (dotted lines, old position and |Δθ|), from '
        '10-ms profile snapshots. (d) F_rest(2 s) in the confirmation set; points, trials; bars, means; '
        'lines pair the activity-only and activity + {u_i} values of each trial; the 16 all-state '
        'values coincide at zero.'
    ),
    'Fig4': (
        'Fig. 4. Route dependence of restoration and the dynamics of the effective synaptic reference '
        'after 11.25° activity-only shifts. (a) F_rest(2 s) in the confirmation set (format as in '
        'Fig. 3(d)); lines pair the output-only and release-only values of each trial. (b) F_rest(t) '
        'with {u_i} profile free (open markers) or clamped (filled markers), from a trailing 150-ms '
        'window; shading, 95% CIs of the clamped curves (16 trials; 15 for the release-only clamp '
        'contrast). (c) Displacements of φ_s, ψ_eff, ψ_u, and ψ_D from the old position in the full '
        'model (means of 16 trials; 10-ms snapshots); lower strip, δ_eff; shading, 95% CIs.'
    ),
}
