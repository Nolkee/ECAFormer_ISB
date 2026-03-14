# Tuning Rounds

This folder organizes three rounds of hyperparameter search:

- round1_s1_s6: first-round single-factor screening.
- round2_s7_s9: second-round combined screening.
- round3_small_search: third-round local search around S8.

## How to run

From project root:

```bash
bash tuning_rounds/round1_s1_s6/run_round1.sh single
bash tuning_rounds/round2_s7_s9/run_round2.sh single
bash tuning_rounds/round3_small_search/run_round3.sh single
```

## Notes

- `results_round1.csv` and `results_round2.csv` are filled with existing best metrics.
- Fill `results_round3_template.csv` after the third-round runs finish.
