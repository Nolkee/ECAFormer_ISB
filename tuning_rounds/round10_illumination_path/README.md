# Round10: Illumination-Path Ablation

This round isolates the two remaining architecture-level bottlenecks in the
ISB branch relative to the baseline illumination path:

1. `illu_map = torch.sigmoid(illu_map)`
2. `x1 = torch.clamp(x_low * illu_map + x_low, 0, 1)` before the denoiser

All three runs inherit the current best stable mainline:

- `Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml`

Only the illumination-path switches change.

## Configs

- `Options/ISB_ecaformer_full_s19_r10_a_x1noclamp.yml`
  - keep illumination sigmoid
  - remove pre-denoiser `x1` clamp

- `Options/ISB_ecaformer_full_s19_r10_b_noillusigmoid.yml`
  - remove illumination sigmoid
  - keep pre-denoiser `x1` clamp

- `Options/ISB_ecaformer_full_s19_r10_c_noillusigmoid_x1noclamp.yml`
  - remove both illumination sigmoid and pre-denoiser `x1` clamp

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/round10_illumination_path/run_round10.sh single
```

## Compare

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ECAFormer_baseline_fair \
         ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4 \
         ISB_ecaformer_full_s19_r10_a_x1noclamp \
         ISB_ecaformer_full_s19_r10_b_noillusigmoid \
         ISB_ecaformer_full_s19_r10_c_noillusigmoid_x1noclamp \
  --out-csv tuning_rounds/round10_illumination_path/results_round10_compare.csv
```
