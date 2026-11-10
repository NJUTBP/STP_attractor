# -*- coding: utf-8 -*-
"""Supplementary-figure legends for the Chinese Physics Letters version (Figs. S1-S10)."""

SUPP_CAPTIONS = {
    1: (
        'Fig. S1. Oscillation diagnostics for the full model and the static and baseline controls. (a) '
        'Mean-subtracted inhibitory currents of the full model and the static control over 5–6.2 s. (b) '
        'Corresponding PSDs; bar, 4–12 Hz. (c) Dominant frequency of the full model, static control and '
        'baseline control in 10 paired trials; side bar, 4–12 Hz. (d) PSD dominant frequency versus the '
        'strongest peak of the autocorrelation function (ACF) at nonzero lag, with identity and '
        'half-frequency guides.'
    ),
    2: (
        'Fig. S2. Frozen-profile control. (a) Time-averaged E→I efficacy profile used in the '
        'frozen-profile control. (b) PSDs of the 10 frozen-profile trials (light) and their mean (dark); '
        'bar, 4–12 Hz. (c) Dominant frequencies of the full model, frozen-profile control and static '
        'control in the 10 paired trials; side bar, 4–12 Hz. All 10 frozen-profile trials passed the '
        'localization criterion.'
    ),
    3: (
        'Fig. S3. Route decomposition of the oscillation frequency across the 42 (τ_f, τ_r) settings '
        '(four trials per setting; outlined cell, reference point; no interpolation). (a)–(c) '
        'Dominant-frequency maps of the full, output-only and release-only models (shared logarithmic '
        'color scale). (d) Fraction of full-model trials with a dominant frequency in the theta band. '
        '(e)–(g) Absolute frequency error relative to the full model of the output-only model, '
        'release-only model and mean-release control. (h) Cumulative distributions of absolute frequency '
        'errors across settings. (i)–(k) Relative calibration residual of the mean efficacy of the same '
        'three controls (numbers, residuals of 2% or more in magnitude). (l) Spike-weighted mean release '
        'fraction ū_rel of the full model, used by the mean-release control.'
    ),
    4: (
        'Fig. S4. Parameter sensitivity within the model. (a) Dominant frequency against the baseline '
        'utilization U at τ_f = 1000 ms and τ_r = 20 ms (four trials per point); dotted line, reference '
        'value; side bar, 4–12 Hz. (b) Absolute frequency error relative to the full model; dashed and '
        'dotted lines, 1 and 2 Hz. (c) F_rest(2 s) after ±11.25° activity-only shifts at U = 0.3 and 0.5 '
        '(filled markers; 8 trials; means and 95% bootstrap CIs) and in the confirmation set at the '
        'reference point (open markers; 16 trials; means). (d) F_rest(2 s) of the full model with τ_f '
        'varied at τ_r = 20 ms or τ_r varied at τ_f = 1000 ms (8 trials per setting; means and 95% '
        'bootstrap CIs).'
    ),
    5: (
        'Fig. S5. Complete results of the confirmation set (16 trials; two shift directions averaged '
        'within trial). (a) Paired contrasts of F_rest(2 s) with 95% bootstrap CIs at 11.25° and 22.5°; '
        'diamonds mark the two prespecified primary contrasts. (b), (c) F_rest(2 s) of the full, '
        'output-only and release-only models and the static control at 11.25° (b) and 22.5° (c); lines '
        'connect output-only and release-only (b) or full and output-only (c) values from the same '
        'trial. (d), (e) F_rest(2 s) after selective state shifts of the full model at 11.25° (d) and '
        '22.5° (e); lines connect activity-only and activity + u values from the same trial.'
    ),
    6: (
        'Fig. S6. Clamp set: positional correction and restoring field with the facilitation profile free or '
        'clamped (16 trials; two shift directions averaged within trial; trailing 150-ms population '
        'vector). (a)–(c) Restoration fraction F_rest(t) after activity-only shifts of 2.8125° to 45° with the '
        'facilitation profile free (solid) or clamped (dashed) in the full (a), output-only (b) and '
        'release-only (c) models (means). (d) F_rest(600 ms) at 11.25° for each trial without and with '
        'the clamp; lines connect values from the same trial. (e) Small-offset gains from per-trial fits '
        'through the origin of clamped restoring velocity against |Δθ| ≤ 16.875° (points, trials; '
        'horizontal marks, means; bars, 95% bootstrap CIs). (f) Paired full-minus-output-only difference '
        'in clamped restoring velocity at each shift magnitude (mean and 95% CI). (g), (h) Clamped '
        'restoring velocity of individual trials (gray) and their mean (colored) in the full (g) and '
        'output-only (h) models. The mean output-only field peaked near 22.5° (31.0° s⁻¹) and was lower '
        'at 33.75° by 13.4° s⁻¹ (95% CI 4.3–22.6); the mean full-model field peaked near 33.75° '
        '(51.5° s⁻¹) and was lower at 45° by 22.2° s⁻¹ (95% CI 3.5–43.5). (i) Release-only restoring '
        'velocity with and without the clamp (mean and 95% CI).'
    ),
    7: (
        'Fig. S7. Profile-snapshot set (16 trials; two shift directions averaged within trial). (a)–(c) '
        'Angular displacement from the pre-shift position of the E-neuron activity center φ_s, effective '
        'synaptic reference ψ_eff, facilitation-profile center ψ_u and depletion-profile center ψ_D after an activity-only '
        'shift in the full model at 22.5° (a), the output-only model at 11.25° (b) and the release-only '
        'model at 11.25° (c); lower strips, δ_eff = d_ax(φ_s, ψ_eff). Lines, means; shading, pointwise '
        '95% bootstrap CIs for φ_s, ψ_eff and δ_eff. The full model at 11.25° is shown in Fig. 4(c) of '
        'the main text.'
    ),
    8: (
        'Fig. S8. Relative-coordinate noise set: 16 trials continued for 10 s at 30 or 60 pA, with two '
        'noise realizations nested within trial (32 runs per condition). (a), (b) Variance across runs '
        'of the effective mismatch δ_eff in the full model at 30 pA (a) and 60 pA (b); logarithmic axis; '
        'light, 20-ms samples; dark, 0.5-s running mean. Late (2–8 s) slopes were 0.0028 deg² s⁻¹ '
        '(95% CI −0.0006 to 0.0060) at 30 pA and 0.0002 deg² s⁻¹ (−0.013 to 0.014) at 60 pA. (c), (d) '
        'MSD of the activity center φ_s, relative to its position at the start of the continuation, for '
        'the full model and, for context only, the static control at 30 pA (c) and 60 pA (d).'
    ),
    9: (
        'Fig. S9. Prespecified noise-robustness test: full model and static control at the reference '
        'point (16 paired trials). (a) Mean absolute displacement at 0, 30, 60 and 90 pA (mean ± SEM '
        'across trials). (b) Noise-integrated displacement B of each trial; lines connect paired '
        'full-model and static-control values; horizontal marks, means. (c) Primary paired '
        'static-minus-full contrast of B with its 95% bootstrap CI.'
    ),
    10: (
        'Fig. S10. E→I efficacy and synaptic-contribution statistics beyond the mean (parameter sweep; '
        'mean-efficacy matching). (a) Differences from the full model at the reference point (means of '
        'four trials) in the temporal coefficient of variation of the efficacy (efficacy CV), the '
        'within-cycle modulation depth of G_j averaged over inhibitory neurons, the release per spike, '
        'the mean weighted E→I synaptic contribution D and the width of the time-averaged c_i profile; '
        'dotted lines, 0 and ±10%. (b) Absolute frequency error of the output-only model against its '
        'absolute difference in modulation depth from the full model at the 42 settings (color, τ_r). '
        '(c) D_model / D_full at the 42 settings (points, settings; bars, medians); dotted line, 1.'
    ),
}

SUPP_CAPTION_VARIANTS = {
    '10_without_p2b': SUPP_CAPTIONS[10],
}
