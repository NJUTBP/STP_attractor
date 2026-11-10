# -*- coding: utf-8 -*-
"""Main-figure legends for the manuscript (Figs. 1-4)."""

CAPTIONS = {
    'Fig1': (
        'Fig. 1. Persistent activity and theta-band oscillations in the full model at the default '
        'parameter setting and σ_η = 90 pA. (a) Network schematic. (b) Spatiotemporal firing pattern '
        'of E neurons (left; black bar indicates cue presentation) and mean firing rate of each E '
        'neuron over 1–8 s (right). Gray bar, the interval expanded in (c), (e), and (f). (c) '
        'Spatiotemporal firing patterns of E (top) and I (bottom) neurons. (d) Power spectral density '
        'for I_inh recorded over 1–8 s, with the primary peak at f_0 = 5.08 Hz and harmonics at 2f_0 '
        'and 3f_0; bar, 4–12 Hz. (e, f) Heat maps showing the temporal evolution of each E neuron’s '
        'facilitation (e) and depletion (f) variables reconstructed from the recorded spike trains. '
        'The color scales start at U and 0, respectively; a square-root color scale is used for D_i.'
    ),
    'Fig2': (
        'Fig. 2. Oscillation frequencies in route-isolated models and their dependence on the release '
        'fraction. (a) Schematic of the output and release routes through which the facilitation '
        'variable affects E→I transmission. (b) Mean-subtracted inhibitory currents from example '
        'trials (common scale bar, 0.5 nA). (c) Dominant frequencies of the output-only and '
        'release-only models plotted against the full-model frequency across 42 (τ_f, τ_r) settings '
        '(means of four trials). Dotted line, identity line; black circles, reference point. (d) '
        'STP-modulated NMDA-mediated contribution, c(t) = u^+ s^NMDA(t)[1 − ρ exp(−t/τ_r)], for a '
        'single E neuron firing at the reference frequency, normalized to its peak value without '
        'depletion. Results are shown for ρ = U and ρ = ū_rel. (e) F(ρ, τ_r, p̄) as a function of ρ '
        'for τ_r = 5, 20, and 80 ms (four trials per point). The triangle and large circle indicate '
        'the output-only and full models, respectively, at the reference point. Vertical bar, '
        '4–12 Hz. (f) Self-consistent frequencies predicted by f = F[u^+(f), τ_r, p̄] plotted against '
        'the corresponding full-model frequencies for the (τ_f, τ_r) sweep (filled markers; symbols '
        'as in e) and the U sweep (open squares). Dotted line, identity line.'
    ),
    'Fig3': (
        'Fig. 3. Positional correction after activity-only and selective STP-state shifts in the full '
        'model. (a–c) Example confirmation trial [black circles in (d)] following shifts of neuronal '
        'activity alone (a), activity and {u_i} (b), or activity and {R_i} (c). Top: Angular extents '
        'above half maximum of the folded NMDA gating variable s^NMDA (blue), facilitation u_i − U '
        '(orange), and resource depletion D_i (green) profiles, measured 10 ms after the shift. '
        'Bottom: Angular displacements of the E-neuron activity center φ_s and facilitation-profile '
        'center ψ_u from the pre-shift position (dotted lines, pre-shift position and |Δθ|), sampled '
        'every 10 ms. (d) F_rest(2 s) in the confirmation set. Points represent individual trials, '
        'and bars indicate means. Lines connect the paired activity-only and activity-plus-u_i '
        'measurements for each trial. The values from all 16 trials in which all state variables were '
        'shifted coincide at zero.'
    ),
    'Fig4': (
        'Fig. 4. Route dependence of positional correction and dynamics of the effective synaptic '
        'reference following 11.25° activity-only shifts. (a) F_rest(2 s) in the confirmation set '
        '(format as in Fig. 3(d)). Lines connect the output-only and release-only values from each '
        'trial. (b) Time courses of F_rest(t) with the {u_i} profile free (open markers) or clamped '
        'at its pre-shift values (filled markers), calculated from spike counts in a trailing 150-ms '
        'window. Shading, 95% CIs for the clamped conditions (16 trials; 15 for the release-only '
        'clamp contrast). (c) Angular displacements of φ_s, ψ_eff, ψ_u, and ψ_D from the pre-shift '
        'position in the full model (means across 16 trials, sampled at 10-ms intervals). The lower '
        'panel shows δ_eff (shading, 95% CIs).'
    ),
}
