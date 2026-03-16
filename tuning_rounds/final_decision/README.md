# Final Decision Pipeline

This folder operationalizes the final 3-step strategy:

1. Lock a stable deliverable from `ISB_ecaformer_full` using **best checkpoint**.
2. Promote `S12` to full training as `Options/ISB_ecaformer_full_s12.yml` and run it.
3. Promote `S18` to full training as `Options/ISB_ecaformer_full_s18.yml` and run it.
4. Promote `S19` to full training as `Options/ISB_ecaformer_full_s19.yml` and run it.
5. Compare `full`, `full_s8`, `full_s12`, `full_s18`, `full_s19` and choose by **best PSNR checkpoint** (not latest).

## Files

- `run_full_s12.sh`: launch full-length training for `full_s12`.
- `run_full_s18.sh`: launch full-length training for `full_s18`.
- `run_full_s19.sh`: launch full-length training for `full_s19`.
- `run_s19_adamw_ablation.sh`: run 3 S19 AdamW ablations sequentially.
- `run_s19_round5_fixes.sh`: run 3 round5 stability/generalization fixes sequentially.
- `compare_full_family.py`: summarize metrics/checkpoints and print ranking.

## Usage

From project root:

```bash
bash tuning_rounds/final_decision/run_full_s12.sh single
bash tuning_rounds/final_decision/run_full_s18.sh single
bash tuning_rounds/final_decision/run_full_s19.sh single
python tuning_rounds/final_decision/compare_full_family.py

# S19 AdamW ablation (single-variable controls):
bash tuning_rounds/final_decision/run_s19_adamw_ablation.sh single

# custom runs:
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ISB_ecaformer_full ISB_ecaformer_full_s8 ISB_ecaformer_full_s12 ISB_ecaformer_full_s18 ISB_ecaformer_full_s19 \
  --out-csv tuning_rounds/final_decision/full_family_compare_with_s18_s19.csv

python tuning_rounds/final_decision/compare_full_family.py \
  --runs ISB_ecaformer_full_s19_adamw_wd1e4 ISB_ecaformer_full_s19_adamw_wd3e4 ISB_ecaformer_full_s19_adamw_wd1e4_lr5e5 \
  --out-csv tuning_rounds/final_decision/s19_adamw_ablation_compare.csv

# Round5 stability/generalization fixes:
bash tuning_rounds/final_decision/run_s19_round5_fixes.sh single

python tuning_rounds/final_decision/compare_full_family.py \
  --runs ISB_ecaformer_full ISB_ecaformer_full_s19 \
         ISB_ecaformer_full_s19_r5_a_wd3e4_tvlow \
         ISB_ecaformer_full_s19_r5_b_identity_softloss \
         ISB_ecaformer_full_s19_r5_c_noamp \
  --out-csv tuning_rounds/final_decision/s19_round5_compare.csv
```

The comparison script writes:

- `tuning_rounds/final_decision/full_family_compare.csv`
