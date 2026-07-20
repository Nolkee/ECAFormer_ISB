Weights live on the server (cavin@192.168.31.20:/home/cavin/Projects/ECAFormer_ISB/experiments/).
Key artifacts:
- ISB champion (logged): r52b @11.5K — reproducible ckpt pending r54 rerun
- Reproducible ours: ISB_ecaformer_r52b_late_ema_anchor_9k/models/net_g_latest.pth [params_ema]
- Baseline: ECAFormer_baseline_lolv1_fair_paper/models/net_g_45000.pth [params_ema] (use_eca=false)
Do NOT commit .pth files or key.pem to git.
