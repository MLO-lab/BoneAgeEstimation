#!/usr/bin/env python
import os
import json
import argparse
from datetime import datetime
import pandas as pd

# New import (clDice-based)
from eval_utils import chunk_based_eval_cldice


def flatten_chunk_eval_cldice(out: dict) -> dict:
    """
    Flatten chunk_based_eval_cldice output into a single row for CSV storage.
    Note: We do NOT grid-search T_cov / T_match here; we run once with defaults
    and store chunk_metrics for later re-labeling / threshold sweeps.
    """
    row = {
        "level": out.get("level", None),
        "t_gt": out["thresholds"]["t_gt"],
        "t_pred": out["thresholds"]["t_pred"],
        "roi_low": out["roi_low"],
        "chunk_alpha": out["chunk_alpha"],
        "do_tolerant": out["do_tolerant"],
        "chunk_z": out["chunk_zyx"][0],
        "chunk_y": out["chunk_zyx"][1],
        "chunk_x": out["chunk_zyx"][2],
    }

    # clDice params (run-time defaults used)
    cp = out.get("cldice_params", {})
    row.update({
        "tau": cp.get("tau", None),
        "T_cov": cp.get("T_cov", None),
        "T_match": cp.get("T_match", None),
        "require_cov_img_for_tp": cp.get("require_cov_img_for_tp", None),
        "use_skeleton_to_skeleton": cp.get("use_skeleton_to_skeleton", None),
    })

    vc = out["voxel_confusion"]
    row.update({
        "voxel_tp": vc["tp"], "voxel_fp": vc["fp"], "voxel_fn": vc["fn"], "voxel_tn": vc["tn"],
        "voxel_precision": vc["precision"], "voxel_recall": vc["recall"], "voxel_f1": vc["f1"],
    })

    pc = out["chunk_presence_confusion"]
    row.update({
        "presence_tp": pc["tp"], "presence_fp": pc["fp"], "presence_fn": pc["fn"], "presence_tn": pc["tn"],
        "presence_precision": pc["precision"], "presence_recall": pc["recall"], "presence_f1": pc["f1"],
    })

    sc = out["chunk_structure_confusion"]
    row.update({
        "struct_tp": sc["tp"], "struct_fp": sc["fp"], "struct_fn": sc["fn"], "struct_tn": sc["tn"],
        "struct_precision": sc["precision"], "struct_recall": sc["recall"], "struct_f1": sc["f1"],
    })

    cl = out["chunk_cldice"]
    row.update({
        "mean_cldice": cl["mean_cldice"],
        "cldice_count": cl["count"],
    })

    acc = out["chunk_accounting"]
    row.update({
        "total_chunks": acc["total_chunks"],
        "used_chunks": acc["used_chunks"],
        "skipped_chunks": acc["skipped_chunks"],
    })

    # How many per-chunk cached rows we saved
    row["n_chunk_metrics_rows"] = len(out.get("chunk_metrics", []))
    row["n_examples_TP"] = len(out.get("examples", {}).get("TP", [])) if "examples" in out else 0
    row["n_examples_FP"] = len(out.get("examples", {}).get("FP", [])) if "examples" in out else 0
    row["n_examples_FN"] = len(out.get("examples", {}).get("FN", [])) if "examples" in out else 0
    row["n_examples_TN"] = len(out.get("examples", {}).get("TN", [])) if "examples" in out else 0

    return row


def parse_args():
    p = argparse.ArgumentParser(description="Run Step3 chunk-based evaluation (clDice) and save results.")
    p.add_argument("--ims_path", type=str, required=True, help="Path to the .ims file.")
    p.add_argument("--level", type=int, default=0, help="Imaris pyramid level.")
    p.add_argument("--ch_gt", type=int, default=2, help="GT channel index (e.g., C3).")
    p.add_argument("--ch_pred", type=int, default=1, help="Pred channel index (e.g., C2).")

    p.add_argument("--t_gt", type=int, required=True, help="Threshold for GT channel.")
    p.add_argument("--t_pred", type=int, required=True, help="Threshold for Pred channel.")

    p.add_argument("--chunk_z", type=int, default=20)
    p.add_argument("--chunk_y", type=int, default=64)
    p.add_argument("--chunk_x", type=int, default=64)

    p.add_argument("--roi_low", type=int, default=1)
    p.add_argument("--chunk_alpha", type=float, default=0.01, help="Chunk presence alpha (fraction).")

    # clDice params (we run once; defaults per your request)
    p.add_argument("--tau", type=float, default=2.0, help="Distance tolerance (in physical units).")
    p.add_argument("--T_cov", type=float, default=0.5, help="Coverage threshold (default 0.5).")
    p.add_argument("--T_match", type=float, default=0.7, help="clDice match threshold (default 0.7).")
    p.add_argument("--require_cov_img_for_tp", action="store_true", help="If set: TP requires cov_img >= T_cov.")
    p.add_argument("--use_skeleton_to_skeleton", action="store_true", help="If set: skeleton-to-skeleton tolerance.")
    p.add_argument("--vz", type=float, default=2.5, help="Voxel size in Z.")
    p.add_argument("--vy", type=float, default=0.7575, help="Voxel size in Y.")
    p.add_argument("--vx", type=float, default=0.7575, help="Voxel size in X.")

    # run configs
    p.add_argument("--do_tolerant", action="store_true", help="If set: tolerant dilation on GT only.")
    p.add_argument("--max_examples_per_class", type=int, default=25)
    p.add_argument("--example_strategy", type=str, default="borderline", choices=["first", "random", "borderline"])

    p.add_argument("--out_dir", type=str, default="results_bone_age", help="Output directory.")
    p.add_argument("--tag", type=str, default="", help="Optional tag added to output filenames.")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    chunk_zyx = (args.chunk_z, args.chunk_y, args.chunk_x)
    voxel_size_zyx = (args.vz, args.vy, args.vx)

    meta = {
        "ims_path": args.ims_path,
        "level": args.level,
        "ch_gt": args.ch_gt,
        "ch_pred": args.ch_pred,
        "t_gt": args.t_gt,
        "t_pred": args.t_pred,
        "chunk_zyx": list(chunk_zyx),
        "roi_low": args.roi_low,
        "chunk_alpha": args.chunk_alpha,
        "cldice": {
            "voxel_size_zyx": list(voxel_size_zyx),
            "tau": args.tau,
            "T_cov": args.T_cov,
            "T_match": args.T_match,
            "require_cov_img_for_tp": bool(args.require_cov_img_for_tp),
            "use_skeleton_to_skeleton": bool(args.use_skeleton_to_skeleton),
        },
        "do_tolerant": bool(args.do_tolerant),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    # ---- run once (no grid over thresholds) ----
    out = chunk_based_eval_cldice(
        ims_path=args.ims_path,
        level=args.level,
        ch_gt=args.ch_gt,
        ch_pred=args.ch_pred,
        t_gt=args.t_gt,
        t_pred=args.t_pred,
        chunk_zyx=chunk_zyx,
        roi_low=args.roi_low,
        chunk_alpha=args.chunk_alpha,
        do_tolerant=bool(args.do_tolerant),

        voxel_size_zyx=voxel_size_zyx,
        tau=float(args.tau),
        T_cov=float(args.T_cov),
        T_match=float(args.T_match),
        require_cov_img_for_tp=bool(args.require_cov_img_for_tp),
        use_skeleton_to_skeleton=bool(args.use_skeleton_to_skeleton),

        hist_bins=20,
        mode="full_eval",
        keep_examples=True,
        max_examples_per_class=int(args.max_examples_per_class),
        example_strategy=args.example_strategy,
        store_chunk_metrics=True,
    )

    # Enrich output for provenance (optional)
    out["level"] = args.level
    out["channels"] = {"ch_gt": args.ch_gt, "ch_pred": args.ch_pred}

    # ---- write outputs ----
    tag = f"_{args.tag}" if args.tag else ""
    base = (
        f"step3_cldice_level{args.level}_tgt{args.t_gt}_tpred{args.t_pred}"
        f"{tag}_chunk{chunk_zyx[0]}x{chunk_zyx[1]}x{chunk_zyx[2]}"
        f"_tau{args.tau}_Tcov{args.T_cov}_Tmatch{args.T_match}"
        f"_tol{int(args.do_tolerant)}"
    )

    # 1) Summary CSV (one row)
    row = flatten_chunk_eval_cldice(out)
    df = pd.DataFrame([row])
    csv_path = os.path.join(args.out_dir, base + "_summary.csv")
    df.to_csv(csv_path, index=False)

    # 2) Full JSON (contains examples + hist + confusions) — can be large
    json_path = os.path.join(args.out_dir, base + ".json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    # 4) Meta
    meta_path = os.path.join(args.out_dir, base + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Saved SUMMARY CSV:", csv_path)
    print("Saved JSON:", json_path)
    print("Saved META:", meta_path)

    # quick summary
    print("\nStructure metrics:")
    print("  struct_f1:", row["struct_f1"])
    print("  struct_precision:", row["struct_precision"])
    print("  struct_recall:", row["struct_recall"])
    print("  mean_cldice:", row["mean_cldice"], " (count:", row["cldice_count"], ")")
    print("  used_chunks:", row["used_chunks"], " skipped:", row["skipped_chunks"])


if __name__ == "__main__":
    main()