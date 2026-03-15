# Round 4 - Three Recovery Groups

This round contains 3 targeted experiments to close the gap vs baseline.

## Configs

- `S17` (Group A: low sigma): reduce bridge stochasticity.
- `S18` (Group B: pixel alignment): increase pixel loss share.
- `S19` (Group C: conservative bridge): lower sigma + lower LR + fewer NFE.

## Run

```bash
bash tuning_rounds/round4_three_groups/run_round4.sh single
```

## Record results

Fill `round4_plan_and_results.csv` after runs finish.
