# NFE Sweep

Use this sweep to evaluate the same checkpoint under multiple inference `nfe`
settings without retraining.

## Default target

- Base config: `Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml`
- Checkpoint: `experiments/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4/best_psnr_20.11_8500.pth`
- Sweep values: `4 6 8 12`

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/final_decision/run_nfe_sweep.sh single
```

## Override checkpoint or output CSV

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/final_decision/run_nfe_sweep.sh single \
  Options/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4.yml \
  experiments/ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4/best_psnr_20.11_8500.pth \
  tuning_rounds/final_decision/results_r6c_nfe_sweep.csv
```

## Outputs

- CSV: `tuning_rounds/final_decision/results_r6c_nfe_sweep.csv`
- Detailed log: `results/nfe_sweep/ISB_ecaformer_full_s19_r6_c_nfe_sweep/sweep.log`
