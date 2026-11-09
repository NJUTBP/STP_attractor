# STP_attractor

Code and figure source data accompanying the manuscript *Separate Routes of Synaptic Facilitation Shape Oscillation Frequency and Bump Restoration in an Attractor Network*.

The model separates the output and release routes of short-term synaptic facilitation at excitatory-to-inhibitory synapses in a spiking continuous-attractor network. This repository contains simulation configurations, analysis code, tests, and the source data used to redraw the manuscript figures.

## Setup

Use Python 3.11. From the repository root, install the tested package versions:

```bash
python -m pip install -r requirements.txt
```

The code was checked with Python 3.11.5. PyMuPDF is used to compose the supplied vector schematic PDFs with the plotted panels.

## Reproduce the figures

```bash
python make_figures/render_all_cpl.py
```

This command redraws main-text Figures 1-4 and Supplementary Figures S1-S10 from the bundled source data, without running new network simulations. It writes 14 PDFs and 14 PNGs to `CPL_Final_Figures/`.

To choose another output folder:

```bash
python make_figures/render_all_cpl.py --output outputs/figures
```

The renderer uses `make_figures/source_data_fig1_noise90/` for Figure 1 by default. Other inputs are in `make_figures/source_data/`, `make_suppfigures/source_data/`, and `make_suppfigures/bundled_data/phase16/`. Some source files retain historical figure numbers; the renderer maps them to the manuscript's current numbering.

## Simulations and tests

Run a short simulation to check the model installation:

```bash
python scripts/run_single.py --config config/smoke.toml --condition dynamic --seed 0 --output results/smoke_run.npz
```

Run the existing model, analysis, and plotting tests:

```bash
python -m pytest tests revision_r1 make_figures/tests -q
```

The tests cover route-specific synaptic updates, checkpoint continuation, activity and synaptic-state shifts, analysis methods, and figure panels. The smoke configuration is a small installation check; manuscript experiment configurations are stored in `config/` and alongside the corresponding experiment modules. Individual scripts expose their options through `--help`.

## Repository contents

| Directory | Contents |
| --- | --- |
| `model/`, `config/` | Network model, synaptic dynamics, and experiment parameters |
| `experiment/`, `scripts/`, `phase16/` | Simulation and intervention protocols |
| `revision_r1/` | Additional route controls, sensitivity analyses, and efficacy statistics |
| `analysis/`, `plot/`, `io_utils/`, `utils/` | Analysis, plotting, and supporting utilities |
| `make_figures/`, `make_suppfigures/` | Manuscript figure scripts, vector schematics, and figure source data |
| `model_sweep_tables/` | Archived parameter-sweep summary tables |
| `tests/` | Model and analysis tests |

Repository: https://github.com/NJUTBP/STP_attractor

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
