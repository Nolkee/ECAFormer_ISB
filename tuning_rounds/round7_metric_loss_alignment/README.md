# Round7: Metric-Loss Alignment (sigma=0.07)

Round7 focuses on controlled verification after metric-hook and loss-path cleanup.

## Fixed baseline conditions (all 3 runs)

- Based on `ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4`
- `sigma_max=0.07`
- `weight_decay=5e-4`
- `train_output_clamp=false`
- `inference_output_clamp=true`
- `strict_output_range=true`
- `loss_on_clamped_output=false`

## Configs

- `Options/ISB_ecaformer_full_s19_r7_a_sigma007.yml`
  - Control group
  - `x0_loss_type=mse`
  - `x0/pixel/tv = 0.4/0.6/0.002`

- `Options/ISB_ecaformer_full_s19_r7_b_sigma007_charb.yml`
  - Replace x0 primary loss with Charbonnier
  - `x0_loss_type=charbonnier`, `x0_charbonnier_eps=1e-3`
  - `x0/pixel/tv = 0.4/0.6/0.002`

- `Options/ISB_ecaformer_full_s19_r7_c_sigma007_charb_x055.yml`
  - Based on B, stronger x0 emphasis
  - `x0/pixel/tv = 0.55/0.45/0.002`

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/round7_metric_loss_alignment/run_round7.sh single
```

## Compare

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ECAFormer_baseline_fair ISB_ecaformer_full_s19 \
         ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4 \
         ISB_ecaformer_full_s19_r7_a_sigma007 \
         ISB_ecaformer_full_s19_r7_b_sigma007_charb \
         ISB_ecaformer_full_s19_r7_c_sigma007_charb_x055 \
  --out-csv tuning_rounds/round7_metric_loss_alignment/results_round7_compare.csv
```
