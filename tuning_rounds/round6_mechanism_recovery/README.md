# Round6: Mechanism Recovery for r5_b

This round focuses on two verified bottlenecks:

1. Hidden training clamp in ISB train path diluted the `identity + softloss` hypothesis.
2. Stochastic reverse noise during inference added metric variance and hurt PSNR consistency.

## Configs

- `Options/ISB_ecaformer_full_s19_r6_a_identity_unclamp_det.yml`
  - Identity output + raw-loss path
  - `train_output_clamp=false`
  - deterministic reverse sampling (`reverse_noise_scale=0.0`)

- `Options/ISB_ecaformer_full_s19_r6_b_identity_unclamp_det_lowlr.yml`
  - Based on A
  - lower lr (`5e-5`) + warmup (`1500`)
  - shorter training budget (`12000`) to reduce late drift

- `Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml`
  - Based on A
  - stronger regularization (`weight_decay=5e-4`)
  - loss rebalance: `x0/pixel/tv = 0.4/0.6/0.002`

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/round6_mechanism_recovery/run_round6.sh single
```

## Compare

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ECAFormer_baseline_fair ISB_ecaformer_full_s19 \
         ISB_ecaformer_full_s19_r5_b_identity_softloss \
         ISB_ecaformer_full_s19_r6_a_identity_unclamp_det \
         ISB_ecaformer_full_s19_r6_b_identity_unclamp_det_lowlr \
         ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4 \
  --out-csv tuning_rounds/round6_mechanism_recovery/results_round6_compare.csv
```
