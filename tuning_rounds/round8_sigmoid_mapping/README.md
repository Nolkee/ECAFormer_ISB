# Round8: Sigmoid Mapping Ablation

This round validates whether bounded output mapping (`output_activation=sigmoid`) can improve stability and perceptual quality without sacrificing PSNR.

## Design rationale

All runs are based on `ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4` (current strong identity baseline), then switch to sigmoid output and keep deterministic reverse sampling.

Fixed settings across R8:
- `output_activation=sigmoid`
- `train_output_clamp=false`
- `inference_output_clamp=false` (sigmoid already bounds to [0,1])
- `reverse_noise_scale=0.0`
- `weight_decay=5e-4`

## Configs

- `Options/ISB_ecaformer_full_s19_r8_a_sigmoid_base.yml`
  - Activation-only control
  - `sigma_max=0.10`, `lr=6e-5`
  - `x0_loss_type=mse`

- `Options/ISB_ecaformer_full_s19_r8_b_sigmoid_charb.yml`
  - Robust x0 loss under sigmoid
  - same as A + `x0_loss_type=charbonnier`, `x0_charbonnier_eps=1e-3`

- `Options/ISB_ecaformer_full_s19_r8_c_sigmoid_charb_sigma007_lowlr.yml`
  - Conservative anti-saturation setup
  - based on B + `sigma_max=0.07`, `lr=5e-5`, `warmup_iter=1500`

## Run

```bash
cd ~/Projects/ECAFormer_ISB
bash tuning_rounds/round8_sigmoid_mapping/run_round8.sh single
```

## Compare

```bash
cd ~/Projects/ECAFormer_ISB
python tuning_rounds/final_decision/compare_full_family.py \
  --runs ECAFormer_baseline_fair \
         ISB_ecaformer_full_s19_r6_c_identity_unclamp_det_wd5e4 \
         ISB_ecaformer_full_s19_r8_a_sigmoid_base \
         ISB_ecaformer_full_s19_r8_b_sigmoid_charb \
         ISB_ecaformer_full_s19_r8_c_sigmoid_charb_sigma007_lowlr \
  --out-csv tuning_rounds/round8_sigmoid_mapping/results_round8_compare.csv
```
