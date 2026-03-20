# Stage A Reference Adaptation

This is the minimal Stage-A implementation derived from the reference
three-stage training idea.

## What it changes

- Starts from pure `ECAFormerBaseline`, not the ISB branch.
- Adds an explicit latent decomposition:
  - `bary_feat`: degradation-agnostic branch
  - `residual_feat`: degradation-related branch
  - `mask_res`: residual gating mask
- Trains with paired LOLv1 data using:
  - pixel reconstruction loss
  - detail-preserving gradient loss
  - barycenter-residual orthogonality loss
  - residual contrastive proxy loss

## What it does not change yet

- no discriminator
- no unpaired data
- no SB transport loss
- no Wasserstein barycenter loss

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/reference_stage_a/run_stage_a.sh single
```

## Compare against the existing baseline

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ECAFormer_baseline_fair ECAFormer_stageA_reference \
  --out-csv tuning_rounds/reference_stage_a/results_stage_a_compare.csv
```
