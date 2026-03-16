# Round5: Stability + Generalization

This round targets two bottlenecks observed in full-S19:

1. Frequent non-finite skip events.
2. Train/Val PSNR gap and overfitting risk.

## Configs

- `Options/ISB_ecaformer_full_s19_r5_a_wd3e4_tvlow.yml`
  - AdamW + `weight_decay=3e-4`
  - `x0_loss_weight=0.3`, `pixel_loss_weight=0.7`, `tv_loss_weight=0.003`
  - `grad_clip_value=0.03`
- `Options/ISB_ecaformer_full_s19_r5_b_identity_softloss.yml`
  - Same as A, but `output_activation=identity`
  - `loss_on_clamped_output=false`
- `Options/ISB_ecaformer_full_s19_r5_c_noamp.yml`
  - Same as A, but `use_amp=false`

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/round5_stability_overfit/run_round5.sh single
```

## Compare

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ISB_ecaformer_full ISB_ecaformer_full_s19 \
         ISB_ecaformer_full_s19_r5_a_wd3e4_tvlow \
         ISB_ecaformer_full_s19_r5_b_identity_softloss \
         ISB_ecaformer_full_s19_r5_c_noamp \
  --out-csv tuning_rounds/round5_stability_overfit/results_round5_compare.csv
```
