# Section 12: Result Aggregation & Visualization Checklist

## Local Implementation

- [x] Defined canonical metrics
  - [x] `NDCG10_off`: Offline NDCG@10 of frozen SASRec (from sasrec_*_stats.json)
  - [x] `CTR_sim`: Synthetic CTR after RLHF epoch 7 (from recrl_*_epoch7_meta.json)
  - [x] `KL_final`: Mean KL divergence in same meta file
  - [x] `Toxic_mean`: Mean toxicity (from safety_report_*)

- [x] Created aggregation script (code/eval/aggregate_main.py)
  - [x] Implemented metric harvesting from JSONs
  - [x] Added CSV generation (docs/main_results.csv)
  - [x] Added LaTeX table generation (docs/main_results.tex)
  - [x] Implemented CTR curve figure generation (docs/fig_ctr_curve.png)
  - [x] Added manifest updates with SHA-256 checksums

- [x] Updated GitHub Actions CI to enforce table completeness
  - [x] Added verification of docs/main_results.csv existence
  - [x] Added check for complete rows and valid CTR values

## RunPod Execution

- [ ] Run aggregation script:
  ```bash
  conda activate rec
  git pull
  python code/eval/aggregate_main.py
  ```

## Manual Verification

- [ ] Check CSV file:
  ```bash
  csvcut -n docs/main_results.csv  # Should show 5 columns
  ```

- [ ] Verify LaTeX compilation:
  - Copy content of docs/main_results.tex into an Overleaf document
  - Confirm no compilation errors

- [ ] Check image properties:
  ```bash
  file docs/fig_ctr_curve.png  # Should report: PNG image data, RGB
  ```

- [ ] Verify manifest updates:
  ```bash
  grep '"type":"result"' data/_checksums/results_manifest.jsonl | wc -l  # Should be ≥ 3
  ```

- [ ] Confirm CI check passes:
  - Push changes to GitHub
  - Check Actions tab for "Results table OK" message

## Manuscript Integration

- [ ] Copy the tabular block from docs/main_results.tex into LaTeX paper section **7.1 Main Table**
- [ ] Insert fig_ctr_curve.png as Figure 3 in the manuscript

## Optional: Interactive Exploration

- [ ] Run Jupyter notebook on RunPod for interactive visualization:
  ```python
  import pandas as pd, plotly.express as px
  df = pd.read_csv("docs/main_results.csv")
  px.bar(df, x="dataset", y="ctr_sim", title="CTR Gain").show()
  ```
