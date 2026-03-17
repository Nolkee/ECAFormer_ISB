#!/usr/bin/env python3
import argparse
import csv
import glob
import os
import re
from typing import Dict, List, Tuple


def parse_metric_csv(path: str) -> List[Tuple[int, float, float, float]]:
    rows = []
    with open(path, "r", newline="") as f:
        for r in csv.reader(f):
            if not r or not r[0].strip().isdigit():
                continue
            try:
                it = int(float(r[0]))
                psnr = float(r[1])
                ssim = float(r[2])
                lpips = float(r[3])
            except Exception:
                continue
            rows.append((it, psnr, ssim, lpips))
    return rows


def _parse_iter_from_best_ckpt(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"best_psnr_[\d.]+_(\d+)\.pth$", base)
    if not m:
        return -1
    return int(m.group(1))


def pick_best_ckpt(exp_dir: str, best_iter: int) -> str:
    cands = sorted(glob.glob(os.path.join(exp_dir, "best_psnr_*.pth")))
    if not cands:
        return ""

    # Prefer exact iteration match from metric.csv (authoritative selection).
    exact = [p for p in cands if _parse_iter_from_best_ckpt(p) == int(best_iter)]
    if exact:
        return sorted(exact)[-1]

    # Fallback: nearest iteration if exact file is missing.
    ranked = sorted(
        cands,
        key=lambda p: (
            abs(_parse_iter_from_best_ckpt(p) - int(best_iter))
            if _parse_iter_from_best_ckpt(p) >= 0 else 10**12,
            -_parse_iter_from_best_ckpt(p),
        ),
    )
    return ranked[0]


def summarize(exp_root: str, run_name: str) -> Dict[str, object]:
    exp_dir = os.path.join(exp_root, run_name)
    metric = os.path.join(exp_dir, "metric.csv")
    row = {
        "run": run_name,
        "status": "missing",
        "best_iter": "",
        "best_psnr": "",
        "best_ssim": "",
        "best_lpips": "",
        "last_iter": "",
        "last_psnr": "",
        "last_ssim": "",
        "last_lpips": "",
        "recommended_ckpt": "",
        "latest_ckpt": os.path.join(exp_dir, "models", "net_g_latest.pth"),
    }

    if not os.path.isfile(metric):
        return row

    recs = parse_metric_csv(metric)
    if not recs:
        row["status"] = "empty"
        return row

    best = max(recs, key=lambda x: x[1])
    last = recs[-1]

    row.update(
        {
            "status": "ok",
            "best_iter": best[0],
            "best_psnr": best[1],
            "best_ssim": best[2],
            "best_lpips": best[3],
            "last_iter": last[0],
            "last_psnr": last[1],
            "last_ssim": last[2],
            "last_lpips": last[3],
            "recommended_ckpt": pick_best_ckpt(exp_dir, best[0]),
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full/full_s8/full_s12 by best metric and checkpoint choice")
    parser.add_argument("--exp-root", default="experiments", help="Experiments root directory")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=["ISB_ecaformer_full", "ISB_ecaformer_full_s8", "ISB_ecaformer_full_s12"],
        help="Run names to compare",
    )
    parser.add_argument(
        "--out-csv",
        default="tuning_rounds/final_decision/full_family_compare.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = [summarize(args.exp_root, n) for n in args.runs]
    ok_rows = [r for r in rows if r["status"] == "ok"]
    ok_rows_sorted = sorted(ok_rows, key=lambda r: (r["best_psnr"], r["best_ssim"], -r["best_lpips"]), reverse=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    fields = [
        "run",
        "status",
        "best_iter",
        "best_psnr",
        "best_ssim",
        "best_lpips",
        "last_iter",
        "last_psnr",
        "last_ssim",
        "last_lpips",
        "recommended_ckpt",
        "latest_ckpt",
    ]

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print("=== Full Family Ranking (by best PSNR) ===")
    for i, r in enumerate(ok_rows_sorted, 1):
        print(
            f"{i}. {r['run']}: best_psnr={r['best_psnr']:.4f}, "
            f"ssim={r['best_ssim']:.4f}, lpips={r['best_lpips']:.4f}, iter={r['best_iter']}"
        )
        if r["recommended_ckpt"]:
            print(f"   use checkpoint: {r['recommended_ckpt']}")
        else:
            print("   use checkpoint: (best file missing, check experiment dir)")

    if ok_rows_sorted:
        top = ok_rows_sorted[0]
        print("\nRecommended final model:")
        print(f"- run: {top['run']}")
        print(f"- checkpoint: {top['recommended_ckpt'] or top['latest_ckpt']}")

    # explicit rule for full_s8
    s8 = next((r for r in ok_rows if r["run"] == "ISB_ecaformer_full_s8"), None)
    if s8:
        print("\nRule reminder for full_s8:")
        print(f"- use BEST: {s8['recommended_ckpt'] or '(best checkpoint not found)'}")
        print(f"- avoid latest: {s8['latest_ckpt']}")

    print(f"\nSaved CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
