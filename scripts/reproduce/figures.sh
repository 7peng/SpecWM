#!/usr/bin/env bash
# Every figure and table of the paper from the tracked CSVs (paper/MANIFEST.md): results/paper/figures/
# (main text) and results/paper/figures/ablations/ (appendix), PNG plus PDF/SVG under vector/.
set -euo pipefail
cd "$(dirname "$0")/../.."; S=results/paper/scripts; F=results/paper/figures; A=$F/ablations
python $S/seed_aggregate.py
python $S/make_table.py                                                        # Table 1
fig() { out=$1; shift; python "$@" --out "$out"; python $S/export_vector.py --dir "$(dirname "$out")/vector" -- "$@" --out "$out"; }
fig $F/fig3_cem.png $S/plot_cem_3env.py                                        # Figure 3
fig $F/fig4_probe.png $S/plot_probe_3env.py                                    # Figure 4
fig $A/fig5_spectral_recovery.png $S/plot_spectral_recovery.py                 # Figure 5
fig $A/fig6_stop_gradient.png $S/plot_anchor_tradeoff.py                       # Figure 6
fig $A/fig7_data_ablation.png $S/plot_play_3env.py --metric both               # Figure 7
fig $A/figB_scene_ablations.png $S/plot_scene_ablations.py --which both         # Figures 8 + 9 in one canvas (not in the paper)
fig $A/fig8_cost_ablation.png $S/plot_scene_ablations.py --which cost          # Figure 8
fig $A/fig9_norm_ablation.png $S/plot_scene_ablations.py --which normalization # Figure 9
python $S/kab_table.py                                                         # centering / tempering ablation
fig $A/figB_centering_tempering.png $S/plot_kab.py
fig $A/figB_centering_tempering_curves.png $S/plot_kab_curves.py
fig $A/figB_centering_tempering_conversion.png $S/kab_conversion.py plot-fade   # spectral conversion through training; data: kab_conversion.py compute --env <env>
fig $A/figB_centering_tempering_heatmap.png $S/kab_conversion.py plot          # the same data as a step x eigenvector heatmap
# Figure 2 reads a checkpoint and one trajectory: python $S/plot_fig2_pipeline.py --out $F/fig2_pipeline.png
