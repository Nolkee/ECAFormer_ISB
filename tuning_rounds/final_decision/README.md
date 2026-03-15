# Final Decision Pipeline

This folder operationalizes the final 3-step strategy:

1. Lock a stable deliverable from `ISB_ecaformer_full` using **best checkpoint**.
2. Promote `S12` to full training as `Options/ISB_ecaformer_full_s12.yml` and run it.
3. Compare `full`, `full_s8`, `full_s12` and choose by **best PSNR checkpoint** (not latest).

## Files

- `run_full_s12.sh`: launch full-length training for `full_s12`.
- `compare_full_family.py`: summarize metrics/checkpoints and print ranking.

## Usage

From project root:

```bash
bash tuning_rounds/final_decision/run_full_s12.sh single
python tuning_rounds/final_decision/compare_full_family.py
```

The comparison script writes:

- `tuning_rounds/final_decision/full_family_compare.csv`

