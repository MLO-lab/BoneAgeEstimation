import h5py
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Any
import time
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from scipy import ndimage as ndi
import random


# -----------------------------
# Reading helpers (Imaris .ims)
# -----------------------------
def open_ims_dataset(ims_path: str, level: int, channel: int):
    """
    Returns an h5py dataset handle for the given channel.
    channel is zero-based if your file uses Channel 0/1/2... (common in IMS).
    """
    f = h5py.File(ims_path, "r")
    ds = f[f"DataSet/ResolutionLevel {level}/TimePoint 0/Channel {channel}/Data"]
    return f, ds  # remember to close f

def get_shape(ims_path: str, level: int, channel: int) -> Tuple[int, int, int]:
    f, ds = open_ims_dataset(ims_path, level, channel)
    shape = tuple(ds.shape)
    f.close()
    return shape

# -----------------------------
# Chunking helpers
# -----------------------------
def iter_chunks_zyx(shape_zyx: Tuple[int, int, int],
                    chunk_zyx: Tuple[int, int, int],
                    start_zyx: Tuple[int, int, int] = (0, 0, 0)) -> Iterable[Tuple[slice, slice, slice]]:
    """
    Yields chunk slices in (Z, Y, X) order.
    """
    Z, Y, X = shape_zyx
    cz, cy, cx = chunk_zyx
    z0, y0, x0 = start_zyx

    for z in range(z0, Z, cz):
        z1 = min(z + cz, Z)
        for y in range(y0, Y, cy):
            y1 = min(y + cy, Y)
            for x in range(x0, X, cx):
                x1 = min(x + cx, X)
                yield (slice(z, z1), slice(y, y1), slice(x, x1))

def read_block(ds, slc_zyx: Tuple[slice, slice, slice]) -> np.ndarray:
    """
    Reads a block (Z,Y,X) and returns numpy array.
    """
    zsl, ysl, xsl = slc_zyx
    # h5py dataset supports ds[zsl, ysl, xsl]
    return ds[zsl, ysl, xsl]

# -----------------------------
# Metrics (continuous "rough")
# -----------------------------
@dataclass
class SoftMetrics:
    soft_iou: float
    soft_dice: float
    spearman_r: Optional[float]  # computed chunkwise approx below (optional)
    pearson_r: Optional[float]

def soft_overlap_metrics(x_u8: np.ndarray, y_u8: np.ndarray, roi: np.ndarray) -> Tuple[float, float]:
    """
    Soft IoU / Soft Dice on normalized [0,1] intensities, within ROI.
    """
    x = (x_u8.astype(np.float32) / 255.0)[roi]
    y = (y_u8.astype(np.float32) / 255.0)[roi]
    if x.size == 0:
        return np.nan, np.nan
    inter = np.minimum(x, y).sum(dtype=np.float64)
    union = np.maximum(x, y).sum(dtype=np.float64)
    soft_iou = inter / union if union > 0 else np.nan
    denom = (x.sum(dtype=np.float64) + y.sum(dtype=np.float64))
    soft_dice = (2.0 * inter / denom) if denom > 0 else np.nan
    return float(soft_iou), float(soft_dice)

def pearson_corr(x_u8: np.ndarray, y_u8: np.ndarray, roi: np.ndarray) -> float:
    x = x_u8.astype(np.float32)[roi]
    y = y_u8.astype(np.float32)[roi]
    if x.size < 2:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.sqrt((x*x).sum()) * np.sqrt((y*y).sum()))
    return float((x*y).sum() / denom) if denom > 0 else np.nan

# Spearman globally chunk-by-chunk is expensive without storing all values.
# If you really need it: sample voxels (e.g., 1e6) and compute spearman with scipy.
# We'll keep it optional.

# -----------------------------
# Voxel confusion matrix
# -----------------------------
@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else np.nan

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else np.nan

    @property
    def f1(self) -> float:
        denom = (2*self.tp + self.fp + self.fn)
        return (2*self.tp) / denom if denom > 0 else np.nan

def update_confusion(conf: Confusion, gt: np.ndarray, pred: np.ndarray) -> Confusion:
    """
    gt/pred are boolean arrays already restricted to ROI.
    """
    tp = int(np.logical_and(gt, pred).sum())
    fp = int(np.logical_and(~gt, pred).sum())
    fn = int(np.logical_and(gt, ~pred).sum())
    tn = int(np.logical_and(~gt, ~pred).sum())
    return Confusion(conf.tp + tp, conf.fp + fp, conf.fn + fn, conf.tn + tn)

# -----------------------------
# Simple 3D dilation (radius=1) without scipy
# -----------------------------
def dilate_3d_radius1(mask: np.ndarray) -> np.ndarray:
    """
    Binary dilation with a 3x3x3 neighborhood (Chebyshev radius 1).
    Pure numpy; for large volumes use chunking + halo (see below).
    """
    m = mask.astype(bool)
    out = m.copy()
    for dz in (-1, 0, 1):
        z_src = slice(max(0, -dz), m.shape[0] - max(0, dz))
        z_dst = slice(max(0, dz),  m.shape[0] - max(0, -dz))
        for dy in (-1, 0, 1):
            y_src = slice(max(0, -dy), m.shape[1] - max(0, dy))
            y_dst = slice(max(0, dy),  m.shape[1] - max(0, -dy))
            for dx in (-1, 0, 1):
                x_src = slice(max(0, -dx), m.shape[2] - max(0, dx))
                x_dst = slice(max(0, dx),  m.shape[2] - max(0, -dx))
                out[z_dst, y_dst, x_dst] |= m[z_src, y_src, x_src]
    return out

# -----------------------------
# Chunk-wise evaluation
# -----------------------------
@dataclass
class ChunkStats:
    # voxel-level confusion accumulated across chunks
    voxel_conf: Confusion
    # chunk-label confusion
    chunk_conf: Confusion
    # chunk IoU summary accumulators
    iou_sum: float
    iou_count: int
    iou_hist: Optional[np.ndarray] = None  # optional histogram

def compute_chunk_label_confusion(gt_bool: np.ndarray, pred_bool: np.ndarray,
                                 alpha: float) -> Tuple[bool, bool]:
    """
    alpha: fraction threshold (e.g., 0.01 means >=1% voxels positive => chunk positive)
    """
    p_gt = gt_bool.mean()
    p_pred = pred_bool.mean()
    return (p_gt >= alpha), (p_pred >= alpha)

def chunk_iou(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    inter = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()
    return float(inter / union) if union > 0 else np.nan

# -----------------------------
# Main pipeline (step-by-step)
# -----------------------------
def evaluate_ims_stepwise(
    ims_path: str,
    level: int,
    ch_c1: int, ch_c2: int, ch_c3: int,
    chunk_zyx: Tuple[int, int, int] = (32, 256, 256),
    roi_low: int = 1,
    # thresholds for voxel-wise binarization
    t2_list: List[int] = None,
    t3_list: List[int] = None,
    t1_list: List[int] = None,
    do_tolerant: bool = True,
    chunk_alpha_list: List[float] = None,
) -> Dict:
    """
    Returns:
      - soft metrics (C1 vs C2, C3 vs C2)
      - threshold sweep results (voxel confusion, strict and tolerant)
      - chunk metrics (chunk-label confusion, chunk IoU stats) for chosen thresholds
    """
    if t2_list is None:
        t2_list = [1,5,10,20]
    if t3_list is None:
        t3_list = [1,5,10,20]
    if t1_list is None:
        t1_list = [1,5,10,20]
    if chunk_alpha_list is None:
        chunk_alpha_list = [0.005, 0.01, 0.02]

    # Open datasets
    f1, ds1 = open_ims_dataset(ims_path, level, ch_c1)
    print(ds1.shape)
    f2, ds2 = open_ims_dataset(ims_path, level, ch_c2)
    f3, ds3 = open_ims_dataset(ims_path, level, ch_c3)

    shape = ds1.shape
    assert ds2.shape == shape and ds3.shape == shape, "Channels must be same shape."

    # ---- Step 1 accumulators (continuous) ----
    soft_inter_12 = 0.0
    soft_union_12 = 0.0
    sumx_12 = 0.0
    sumy_12 = 0.0
    # for soft dice, we need sum(min), sum(x), sum(y)
    soft_inter_32 = 0.0
    soft_union_32 = 0.0
    sumx_32 = 0.0
    sumy_32 = 0.0

    # Pearson accumulators (streaming)
    # We'll compute Pearson on ROI by accumulating sums.
    # pearson = cov / (stdx*stdy). Need sum(x), sum(y), sum(x^2), sum(y^2), sum(xy), n
    def init_corr_acc():
        return dict(n=0, sx=0.0, sy=0.0, sxx=0.0, syy=0.0, sxy=0.0)

    corr12 = init_corr_acc()
    corr32 = init_corr_acc()

    # ---- Step 2 accumulators (threshold sweeps) ----
    # We store confusion matrices for each (t_gt, t_pred) pair.
    # For C3-derived GT vs C2:
    voxel_conf_3v2 = {(t3, t2): Confusion(0,0,0,0) for t3 in t3_list for t2 in t2_list}
    voxel_conf_1v2 = {(t1, t2): Confusion(0,0,0,0) for t1 in t1_list for t2 in t2_list}

    # Tolerant versions (GT dilated by radius 1 inside chunk with halo handled later; here we do per-chunk local dilation)
    voxel_conf_3v2_tol = {(t3, t2): Confusion(0,0,0,0) for t3 in t3_list for t2 in t2_list} if do_tolerant else None
    voxel_conf_1v2_tol = {(t1, t2): Confusion(0,0,0,0) for t1 in t1_list for t2 in t2_list} if do_tolerant else None

    # We'll do tolerant dilation per chunk by reading a halo of 1 voxel around the chunk.
    halo = 1 if do_tolerant else 0

    # ---- Iterate chunks ----
    for slc in iter_chunks_zyx(shape, chunk_zyx):
        zsl, ysl, xsl = slc

        # Halo slices for tolerant dilation (clamped to volume bounds)
        if halo > 0:
            z0, z1 = zsl.start, zsl.stop
            y0, y1 = ysl.start, ysl.stop
            x0, x1 = xsl.start, xsl.stop
            zsl_h = slice(max(0, z0-halo), min(shape[0], z1+halo))
            ysl_h = slice(max(0, y0-halo), min(shape[1], y1+halo))
            xsl_h = slice(max(0, x0-halo), min(shape[2], x1+halo))
            slc_h = (zsl_h, ysl_h, xsl_h)
        else:
            slc_h = slc

        c1 = read_block(ds1, slc_h).astype(np.uint8)
        c2 = read_block(ds2, slc_h).astype(np.uint8)
        c3 = read_block(ds3, slc_h).astype(np.uint8)

        # ROI for this halo-block
        roi_h = (c1 >= roi_low) | (c2 >= roi_low) | (c3 >= roi_low)
        if roi_h.sum() == 0:
            continue

        # -------------------
        # Step 1: continuous
        # -------------------
        # Use only the inner (non-halo) part for consistency
        if halo > 0:
            # compute inner indices within the halo-block
            iz0 = zsl.start - slc_h[0].start
            iy0 = ysl.start - slc_h[1].start
            ix0 = xsl.start - slc_h[2].start
            iz1 = iz0 + (zsl.stop - zsl.start)
            iy1 = iy0 + (ysl.stop - ysl.start)
            ix1 = ix0 + (xsl.stop - xsl.start)
            inner = (slice(iz0, iz1), slice(iy0, iy1), slice(ix0, ix1))
        else:
            inner = (slice(None), slice(None), slice(None))

        c1i = c1[inner]; c2i = c2[inner]; c3i = c3[inner]
        roii = roi_h[inner]

        # Soft overlaps on normalized [0,1]
        x12 = (c1i.astype(np.float32) / 255.0)[roii]
        y12 = (c2i.astype(np.float32) / 255.0)[roii]
        if x12.size > 0:
            soft_inter_12 += float(np.minimum(x12, y12).sum(dtype=np.float64))
            soft_union_12 += float(np.maximum(x12, y12).sum(dtype=np.float64))
            sumx_12 += float(x12.sum(dtype=np.float64))
            sumy_12 += float(y12.sum(dtype=np.float64))

            # Pearson accumulators
            corr12["n"] += int(x12.size)
            corr12["sx"] += float(x12.sum(dtype=np.float64))
            corr12["sy"] += float(y12.sum(dtype=np.float64))
            corr12["sxx"] += float((x12*x12).sum(dtype=np.float64))
            corr12["syy"] += float((y12*y12).sum(dtype=np.float64))
            corr12["sxy"] += float((x12*y12).sum(dtype=np.float64))

        x32 = (c3i.astype(np.float32) / 255.0)[roii]
        y32 = (c2i.astype(np.float32) / 255.0)[roii]
        if x32.size > 0:
            soft_inter_32 += float(np.minimum(x32, y32).sum(dtype=np.float64))
            soft_union_32 += float(np.maximum(x32, y32).sum(dtype=np.float64))
            sumx_32 += float(x32.sum(dtype=np.float64))
            sumy_32 += float(y32.sum(dtype=np.float64))

            corr32["n"] += int(x32.size)
            corr32["sx"] += float(x32.sum(dtype=np.float64))
            corr32["sy"] += float(y32.sum(dtype=np.float64))
            corr32["sxx"] += float((x32*x32).sum(dtype=np.float64))
            corr32["syy"] += float((y32*y32).sum(dtype=np.float64))
            corr32["sxy"] += float((x32*y32).sum(dtype=np.float64))

        # ----------------------------
        # Step 2: voxel confusion sweep
        # ----------------------------
        # Work inside inner region ROI only (drop halo)
        # We'll build masks once per threshold.
        # C2 pred candidates:
        pred2 = {t2: (c2i >= t2) & roii for t2 in t2_list}
        # GT from C3 proxy:
        gt3 = {t3: (c3i >= t3) & roii for t3 in t3_list}
        # GT from C1 (CD105 signal):
        gt1 = {t1: (c1i >= t1) & roii for t1 in t1_list}

        if do_tolerant:
            # Need dilation on GT in the INNER region, but dilation must consider neighborhood.
            # Easiest: dilate on halo-block, then crop to inner.
            # We'll compute gt masks on halo-block first, dilate, then crop inner.
            # Pred stays inner.
            pred2_inner = pred2

            # gt3 on halo-block
            gt3_h = {t3: (c3 >= t3) & roi_h for t3 in t3_list}
            gt1_h = {t1: (c1 >= t1) & roi_h for t1 in t1_list}

            gt3_tol = {}
            gt1_tol = {}
            for t3 in t3_list:
                dil = dilate_3d_radius1(gt3_h[t3])
                gt3_tol[t3] = dil[inner]  # crop to inner
            for t1 in t1_list:
                dil = dilate_3d_radius1(gt1_h[t1])
                gt1_tol[t1] = dil[inner]
        else:
            gt3_tol = None
            gt1_tol = None

        # Update confusion matrices
        for t2, pred in pred2.items():
            # C3 vs C2
            for t3, gt in gt3.items():
                voxel_conf_3v2[(t3, t2)] = update_confusion(voxel_conf_3v2[(t3, t2)], gt, pred)
                if do_tolerant:
                    voxel_conf_3v2_tol[(t3, t2)] = update_confusion(voxel_conf_3v2_tol[(t3, t2)], gt3_tol[t3] & roii, pred)

            # C1 vs C2
            for t1, gt in gt1.items():
                voxel_conf_1v2[(t1, t2)] = update_confusion(voxel_conf_1v2[(t1, t2)], gt, pred)
                if do_tolerant:
                    voxel_conf_1v2_tol[(t1, t2)] = update_confusion(voxel_conf_1v2_tol[(t1, t2)], gt1_tol[t1] & roii, pred)

    # Close files
    f1.close(); f2.close(); f3.close()

    # Finalize Step 1 metrics
    soft_iou_12 = soft_inter_12 / soft_union_12 if soft_union_12 > 0 else np.nan
    soft_dice_12 = (2*soft_inter_12 / (sumx_12 + sumy_12)) if (sumx_12 + sumy_12) > 0 else np.nan

    soft_iou_32 = soft_inter_32 / soft_union_32 if soft_union_32 > 0 else np.nan
    soft_dice_32 = (2*soft_inter_32 / (sumx_32 + sumy_32)) if (sumx_32 + sumy_32) > 0 else np.nan

    def finalize_pearson(acc):
        n = acc["n"]
        if n < 2:
            return np.nan
        sx, sy, sxx, syy, sxy = acc["sx"], acc["sy"], acc["sxx"], acc["syy"], acc["sxy"]
        mx = sx / n
        my = sy / n
        cov = (sxy / n) - mx*my
        vx = (sxx / n) - mx*mx
        vy = (syy / n) - my*my
        denom = np.sqrt(vx) * np.sqrt(vy)
        return float(cov / denom) if denom > 0 else np.nan

    pearson12 = finalize_pearson(corr12)
    pearson32 = finalize_pearson(corr32)

    # Convert sweep confusions to result tables (dicts)
    def sweep_to_list(conf_dict: Dict[Tuple[int,int], Confusion], label: str):
        rows = []
        for (tgt, tpred), c in conf_dict.items():
            rows.append(dict(
                comparison=label,
                t_gt=int(tgt),
                t_pred=int(tpred),
                tp=c.tp, fp=c.fp, fn=c.fn, tn=c.tn,
                precision=c.precision, recall=c.recall, f1=c.f1
            ))
        # sort by best f1 desc
        rows.sort(key=lambda r: (-(r["f1"] if np.isfinite(r["f1"]) else -1e9)))
        return rows

    results = {
        "step1_soft": {
            "C1_vs_C2": {"soft_iou": float(soft_iou_12), "soft_dice": float(soft_dice_12), "pearson": float(pearson12)},
            "C3_vs_C2": {"soft_iou": float(soft_iou_32), "soft_dice": float(soft_dice_32), "pearson": float(pearson32)},
        },
        "step2_sweeps": {
            "C3_vs_C2_strict": sweep_to_list(voxel_conf_3v2, "C3_vs_C2_strict"),
            "C1_vs_C2_strict": sweep_to_list(voxel_conf_1v2, "C1_vs_C2_strict"),
        }
    }
    if do_tolerant:
        results["step2_sweeps"]["C3_vs_C2_tol"] = sweep_to_list(voxel_conf_3v2_tol, "C3_vs_C2_tol")
        results["step2_sweeps"]["C1_vs_C2_tol"] = sweep_to_list(voxel_conf_1v2_tol, "C1_vs_C2_tol")

    # -----------------------------
    # Step 3: chunk-based metrics
    # -----------------------------
    # For step 3, pick ONE threshold pair (or a few) from the sweep.
    # We'll implement a function you can call after you choose thresholds.
    results["step3_helper_note"] = (
        "Use choose_thresholds_from_sweep(...) to pick (t_gt,t_pred), then run chunk_based_eval(...) below."
    )
    results["recommended_chunk_size_vox"] = {
        "example_50um_cube": {"chunk_zyx": (20, 64, 64), "note": "≈50µm in Z and ≈48.5µm in XY"}
    }
    results["voxel_size_um"] = {"x": 0.7575, "y": 0.7575, "z": 2.5}
    results["chunk_alpha_candidates"] = chunk_alpha_list

    return results

# -----------------------------
# Utility: pick thresholds from sweep output
# -----------------------------
def choose_thresholds_from_sweep(sweep_rows: List[Dict], topk: int = 10) -> List[Dict]:
    """
    Return the top-k rows by F1 from the sweep result list.
    """
    return sweep_rows[:topk]


# The final version of the chunk-based eval with tolerant mode and also the overlap_iou_threshold parameter
# -----------------------------
def chunk_based_eval(
    ims_path: str,
    level: int,
    ch_gt: int,
    ch_pred: int,
    t_gt: int,
    t_pred: int,
    chunk_zyx: Tuple[int, int, int] = (20, 64, 64),
    roi_low: int = 1,
    chunk_alpha: float = 0.01,
    overlap_iou_threshold: float = 0.1,
    hist_bins: int = 20,
    do_tolerant: bool = False,
) -> Dict:
    """
    Chunk-based evaluation between:
      GT   = (Channel ch_gt   >= t_gt)
      Pred = (Channel ch_pred >= t_pred)
    within ROI = union-of-signal >= roi_low.

    Tolerant mode (do_tolerant=True):
      - Apply 3D dilation (radius=1 voxel) to the GT binary mask only.
      - Pred is not dilated.
      - This mirrors Step2 tolerant behavior.

    Returns:
      - voxel confusion
      - chunk presence confusion (baseline, alpha fraction)
      - chunk overlap confusion (requires IoU >= overlap_iou_threshold on GT-positive chunks)
      - chunk IoU histograms (all used chunks and GT-positive chunks)
      - chunk accounting
    """
    f_gt, ds_gt = open_ims_dataset(ims_path, level, ch_gt)
    f_pr, ds_pr = open_ims_dataset(ims_path, level, ch_pred)
    shape = ds_gt.shape
    assert ds_pr.shape == shape, "GT and Pred channels must have the same shape."

    voxel_conf = Confusion(0, 0, 0, 0)
    chunk_presence_conf = Confusion(0, 0, 0, 0)
    chunk_overlap_conf = Confusion(0, 0, 0, 0)

    iou_hist_all = np.zeros(hist_bins, dtype=np.int64)
    iou_sum_all = 0.0
    iou_count_all = 0

    iou_hist_gtpos = np.zeros(hist_bins, dtype=np.int64)
    iou_sum_gtpos = 0.0
    iou_count_gtpos = 0

    total_chunks = 0
    used_chunks = 0
    skipped_chunks = 0

    # In tolerant mode we need a halo so dilation near chunk borders is consistent
    halo = 1 if do_tolerant else 0

    for slc in iter_chunks_zyx(shape, chunk_zyx):
        total_chunks += 1

        # Build halo slice (clamped)
        zsl, ysl, xsl = slc
        z0 = max(0, zsl.start - halo); z1 = min(shape[0], zsl.stop + halo)
        y0 = max(0, ysl.start - halo); y1 = min(shape[1], ysl.stop + halo)
        x0 = max(0, xsl.start - halo); x1 = min(shape[2], xsl.stop + halo)
        slc_h = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        gt_h = read_block(ds_gt, slc_h).astype(np.uint8)
        pr_h = read_block(ds_pr, slc_h).astype(np.uint8)

        # ROI defined by union of signals (only these two channels)
        roi_h = (gt_h >= roi_low) | (pr_h >= roi_low)
        if roi_h.sum() == 0:
            skipped_chunks += 1
            continue
        used_chunks += 1

        # Binarize within ROI (halo block)
        gt_b_h = (gt_h >= t_gt) & roi_h
        pr_b_h = (pr_h >= t_pred) & roi_h

        # Apply tolerant dilation to GT only (halo block), then crop back to inner chunk
        if do_tolerant:
            gt_b_h = dilate_3d_radius1(gt_b_h)

        # Crop to inner chunk coordinates inside the halo block
        iz0 = zsl.start - z0; iz1 = iz0 + (zsl.stop - zsl.start)
        iy0 = ysl.start - y0; iy1 = iy0 + (ysl.stop - ysl.start)
        ix0 = xsl.start - x0; ix1 = ix0 + (xsl.stop - xsl.start)

        gt_b = gt_b_h[iz0:iz1, iy0:iy1, ix0:ix1]
        pr_b = pr_b_h[iz0:iz1, iy0:iy1, ix0:ix1]

        # Voxel-level confusion (inner chunk)
        voxel_conf = update_confusion(voxel_conf, gt_b, pr_b)

        # Chunk presence labels (baseline)
        gt_pos, pr_pos = compute_chunk_label_confusion(gt_b, pr_b, alpha=chunk_alpha)

        chunk_presence_conf = update_confusion(
            chunk_presence_conf,
            np.array([gt_pos], dtype=bool),
            np.array([pr_pos], dtype=bool),
        )

        # Chunk IoU (binary, inner chunk)
        iou = chunk_iou(gt_b, pr_b)
        if np.isfinite(iou):
            iou_sum_all += iou
            iou_count_all += 1
            b_all = min(hist_bins - 1, int(iou * hist_bins))
            iou_hist_all[b_all] += 1

            if gt_pos:
                iou_sum_gtpos += iou
                iou_count_gtpos += 1
                b_gt = min(hist_bins - 1, int(iou * hist_bins))
                iou_hist_gtpos[b_gt] += 1

        # Overlap-based prediction label:
        # - If GT chunk is positive: require both presence and IoU >= threshold
        # - If GT chunk is negative: keep pr_pos to penalize false alarms
        if gt_pos:
            pr_overlap_pos = bool(pr_pos and np.isfinite(iou) and (iou >= overlap_iou_threshold))
        else:
            pr_overlap_pos = bool(pr_pos)

        chunk_overlap_conf = update_confusion(
            chunk_overlap_conf,
            np.array([gt_pos], dtype=bool),
            np.array([pr_overlap_pos], dtype=bool),
        )

    f_gt.close()
    f_pr.close()

    print(f"Total chunks: {total_chunks}, Used chunks: {used_chunks}, Skipped chunks: {skipped_chunks}")

    return {
        "thresholds": {"t_gt": int(t_gt), "t_pred": int(t_pred)},
        "chunk_zyx": chunk_zyx,
        "roi_low": int(roi_low),
        "chunk_alpha": float(chunk_alpha),
        "overlap_iou_threshold": float(overlap_iou_threshold),
        "do_tolerant": bool(do_tolerant),

        "voxel_confusion": {
            "tp": voxel_conf.tp, "fp": voxel_conf.fp, "fn": voxel_conf.fn, "tn": voxel_conf.tn,
            "precision": voxel_conf.precision, "recall": voxel_conf.recall, "f1": voxel_conf.f1,
        },

        "chunk_presence_confusion": {
            "tp": chunk_presence_conf.tp, "fp": chunk_presence_conf.fp,
            "fn": chunk_presence_conf.fn, "tn": chunk_presence_conf.tn,
            "precision": chunk_presence_conf.precision,
            "recall": chunk_presence_conf.recall,
            "f1": chunk_presence_conf.f1,
        },

        "chunk_overlap_confusion": {
            "tp": chunk_overlap_conf.tp, "fp": chunk_overlap_conf.fp,
            "fn": chunk_overlap_conf.fn, "tn": chunk_overlap_conf.tn,
            "precision": chunk_overlap_conf.precision,
            "recall": chunk_overlap_conf.recall,
            "f1": chunk_overlap_conf.f1,
        },

        "chunk_iou": {
            "mean_iou": (iou_sum_all / iou_count_all) if iou_count_all > 0 else np.nan,
            "count": int(iou_count_all),
            "hist_bins": int(hist_bins),
            "hist": iou_hist_all.tolist(),
        },

        "chunk_iou_on_gt_positive": {
            "mean_iou": (iou_sum_gtpos / iou_count_gtpos) if iou_count_gtpos > 0 else np.nan,
            "count": int(iou_count_gtpos),
            "hist_bins": int(hist_bins),
            "hist": iou_hist_gtpos.tolist(),
        },

        "chunk_accounting": {
            "total_chunks": int(total_chunks),
            "used_chunks": int(used_chunks),
            "skipped_chunks": int(skipped_chunks),
        },
    }
    

EPS = 1e-8

def _safe_skeletonize(mask: np.ndarray) -> np.ndarray:
    """
    Purpose:
      Produce a 1-voxel-thick skeleton (centerline) from a binary mask.
      Works for 2D or 3D because skimage.morphology.skeletonize supports nD.

    Input:
      mask: np.ndarray
        - binary-like array (bool or 0/1)
        - shape: (Z,Y,X) for 3D chunks

    Output:
      sk: np.ndarray (bool)
        - same shape as mask
        - True voxels represent the skeleton.
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=bool)
    return skeletonize(m).astype(bool)

def _edt_distance_to_mask(mask: np.ndarray, voxel_size_zyx: Tuple[float, float, float]) -> np.ndarray:
    """
    Purpose:
      Compute, for every voxel, the distance to the nearest True voxel in `mask`.
      Distances are returned in physical units if voxel_size_zyx is set in µm.

    Input:
      mask: bool array, True = target structure
      voxel_size_zyx: (vz, vy, vx)
        - sampling for each axis
        - if you pass (1,1,1), distances are in voxels
        - if you pass real spacing (e.g. (2.5, 0.7575, 0.7575)), distances are in µm

    Output:
      dist: float array, same shape
        dist[p] = distance from voxel p to nearest True voxel in `mask`
    """
    m = mask.astype(bool)
    # EDT gives distance to nearest zero; invert so that distance is to True voxels.
    return distance_transform_edt(~m, sampling=voxel_size_zyx)


def skeleton_coverages_and_cldice(
    gt_mask: np.ndarray,
    pr_mask: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    tau: float = 2.0,
    use_skeleton_to_skeleton: bool = False,
) -> Tuple[float, float, float, Dict]:
    """
    Purpose:
      Compute structure-wise agreement between GT and Pred in a way that is robust to small shifts.

    What it returns:
      cov_img  (recall-like):
        fraction of GT skeleton points that lie within tau of Pred structure
      cov_pred (precision-like):
        fraction of Pred skeleton points that lie within tau of GT structure
      clDice:
        combines cov_img and cov_pred (high only if both are high)
      debug:
        sizes of masks/skeletons for sanity checking

    Parameters:
      gt_mask, pr_mask:
        binary masks (bool or 0/1) of the structure in one chunk.

      voxel_size_zyx:
        spacing (vz, vy, vx). Controls the units of tau.
        - if you pass real spacing in µm, tau is in µm.
        - if you pass (1,1,1), tau is in voxels.

      tau:
        tolerance distance.
        A skeleton voxel is considered matched if it's within tau of the other structure.

      use_skeleton_to_skeleton:
        - False (recommended): skeleton points match against the other *mask*.
          More forgiving when thickness differs (common in sinusoids).
        - True: skeleton points match against the other *skeleton*.
          Stricter; punishes thickness differences less relevant to centerline accuracy.

    Notes on edge cases:
      - If one skeleton is empty, coverage is 1 if the other skeleton is also empty, else 0.
    """
    gt = gt_mask.astype(bool)
    pr = pr_mask.astype(bool)

    gt_sk = _safe_skeletonize(gt)
    pr_sk = _safe_skeletonize(pr)

    n_gt_sk = int(gt_sk.sum())
    n_pr_sk = int(pr_sk.sum())

    pr_target = pr_sk if use_skeleton_to_skeleton else pr
    gt_target = gt_sk if use_skeleton_to_skeleton else gt

    # cov_img: GT skeleton supported by Pred target
    if pr_target.sum() == 0:
        cov_img = 1.0 if n_gt_sk == 0 else 0.0
    else:
        dist_to_pr = _edt_distance_to_mask(pr_target, voxel_size_zyx)
        cov_img = 1.0 if n_gt_sk == 0 else float((dist_to_pr[gt_sk] <= tau).mean())

    # cov_pred: Pred skeleton supported by GT target
    if gt_target.sum() == 0:
        cov_pred = 1.0 if n_pr_sk == 0 else 0.0
    else:
        dist_to_gt = _edt_distance_to_mask(gt_target, voxel_size_zyx)
        cov_pred = 1.0 if n_pr_sk == 0 else float((dist_to_gt[pr_sk] <= tau).mean())

    # clDice combines both
    cldice = 0.0 if (cov_img + cov_pred) < EPS else float(2.0 * cov_img * cov_pred / (cov_img + cov_pred + EPS))

    debug = {
        "n_gt": int(gt.sum()),
        "n_pr": int(pr.sum()),
        "n_gt_sk": n_gt_sk,
        "n_pr_sk": n_pr_sk,
    }
    return cov_img, cov_pred, cldice, debug


# --- example selection helper ---
def _example_priority(
    label: str,
    cov_img: float,
    cov_pred: float,
    cldice: float,
    T_match: float,
    T_cov: float,
) -> float:
    if label == "TP":
        return min(abs(cldice - T_match), abs(cov_img - T_cov))
    if label == "FN":
        return abs(cov_img - T_cov)
    if label == "FP":
        return abs(cov_pred - T_cov)
    return 0.0


def _maybe_add_example(
    examples: Dict[str, list],
    label: str,
    ex: "ChunkExample",
    max_examples_per_class: int,
    example_strategy: str,
    priority: float,
):
    lst = examples[label]
    if len(lst) < max_examples_per_class:
        lst.append((priority, ex))
        return

    if example_strategy == "first":
        return

    if example_strategy == "random":
        j = random.randrange(max_examples_per_class)
        lst[j] = (priority, ex)
        return

    if example_strategy == "borderline":
        worst_idx = max(range(len(lst)), key=lambda i: lst[i][0])
        if priority < lst[worst_idx][0]:
            lst[worst_idx] = (priority, ex)
        return

    raise ValueError(f"Unknown example_strategy={example_strategy}")


@dataclass
class ChunkExample:
    slc: Tuple[slice, slice, slice]
    has_img: bool
    has_pred: bool
    label: str
    cov_img: float
    cov_pred: float
    cldice: float
    GTvox: int
    PRvox: int
    GTsk: int
    PRsk: int


def chunk_based_eval_cldice(
    ims_path: str,
    level: int,
    ch_gt: int,
    ch_pred: int,
    t_gt: int,
    t_pred: int,
    chunk_zyx: Tuple[int, int, int] = (20, 64, 64),
    roi_low: int = 1,
    chunk_alpha: float = 0.01,
    do_tolerant: bool = False,

    # clDice params
    voxel_size_zyx: Tuple[float, float, float] = (2.5, 0.7575, 0.7575),
    tau: float = 2.0,
    T_match: float = 0.7,
    T_cov: float = 0.6,
    require_cov_img_for_tp: bool = True,
    use_skeleton_to_skeleton: bool = False,

    hist_bins: int = 20,

    # defaults: full eval + keep examples
    mode: str = "full_eval",          # "examples_only" or "full_eval"
    keep_examples: bool = True,
    max_examples_per_class: int = 25,
    example_strategy: str = "borderline",

    # caching: store per-chunk metrics for later threshold sweeps
    store_chunk_metrics: bool = True,

    # early-stop only for examples_only
    examples_need: Tuple[str, ...] = ("TP", "FP", "FN"),
) -> Dict:
    if mode not in {"examples_only", "full_eval"}:
        raise ValueError(f"mode must be 'examples_only' or 'full_eval', got {mode}")

    f_gt, ds_gt = open_ims_dataset(ims_path, level, ch_gt)
    f_pr, ds_pr = open_ims_dataset(ims_path, level, ch_pred)
    shape = ds_gt.shape
    assert ds_pr.shape == shape, "GT and Pred channels must have the same shape."

    voxel_conf = Confusion(0, 0, 0, 0)
    chunk_presence_conf = Confusion(0, 0, 0, 0)
    chunk_struct_conf = Confusion(0, 0, 0, 0)

    cldice_hist_all = np.zeros(hist_bins, dtype=np.int64)
    cldice_sum_all = 0.0
    cldice_count_all = 0

    cov_img_hist = np.zeros(hist_bins, dtype=np.int64)
    cov_pred_hist = np.zeros(hist_bins, dtype=np.int64)

    total_chunks = 0
    used_chunks = 0
    skipped_chunks = 0

    halo = 1 if do_tolerant else 0
    examples = {"TP": [], "FP": [], "FN": [], "TN": []}
    chunk_metrics: List[dict] = []

    def _slc_to_coords(s: slice):
        return int(s.start), int(s.stop)

    for slc in iter_chunks_zyx(shape, chunk_zyx):
        total_chunks += 1

        zsl, ysl, xsl = slc
        z0 = max(0, zsl.start - halo); z1 = min(shape[0], zsl.stop + halo)
        y0 = max(0, ysl.start - halo); y1 = min(shape[1], ysl.stop + halo)
        x0 = max(0, xsl.start - halo); x1 = min(shape[2], xsl.stop + halo)
        slc_h = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

        gt_h = read_block(ds_gt, slc_h).astype(np.uint8)
        pr_h = read_block(ds_pr, slc_h).astype(np.uint8)

        roi_h = (gt_h >= roi_low) | (pr_h >= roi_low)
        if roi_h.sum() == 0:
            skipped_chunks += 1
            continue
        used_chunks += 1

        gt_b_h = (gt_h >= t_gt) & roi_h
        pr_b_h = (pr_h >= t_pred) & roi_h

        if do_tolerant:
            gt_b_h = dilate_3d_radius1(gt_b_h)

        iz0 = zsl.start - z0; iz1 = iz0 + (zsl.stop - zsl.start)
        iy0 = ysl.start - y0; iy1 = iy0 + (ysl.stop - ysl.start)
        ix0 = xsl.start - x0; ix1 = ix0 + (xsl.stop - xsl.start)

        gt_b = gt_b_h[iz0:iz1, iy0:iy1, ix0:ix1]
        pr_b = pr_b_h[iz0:iz1, iy0:iy1, ix0:ix1]

        # counts that are always cheap
        GTvox = int(gt_b.sum())
        PRvox = int(pr_b.sum())

        # presence
        gt_pos, pr_pos = compute_chunk_label_confusion(gt_b, pr_b, alpha=chunk_alpha)
        has_img, has_pred = bool(gt_pos), bool(pr_pos)

        # defaults
        cov_img = cov_pred = cldice = np.nan
        GTsk = 0 if GTvox == 0 else -1
        PRsk = 0 if PRvox == 0 else -1

        # label by presence when not both present
        # if not (has_img and has_pred):
        if (not has_img) and (not has_pred):
            label = "TN"
            # elif has_img and (not has_pred):
            #     label = "FN"
            # elif (not has_img) and has_pred:
            #     label = "FP"
        else:
            cov_img, cov_pred, cldice, dbg = skeleton_coverages_and_cldice(
                gt_mask=gt_b,
                pr_mask=pr_b,
                voxel_size_zyx=voxel_size_zyx,
                tau=tau,
                use_skeleton_to_skeleton=use_skeleton_to_skeleton,
            )

            # Pull skeleton counts from dbg if provided
            if isinstance(dbg, dict):
                GTsk = int(dbg.get("GTsk", GTsk))
                PRsk = int(dbg.get("PRsk", PRsk))

            tp_cond = (cldice >= T_match)
            if require_cov_img_for_tp:
                tp_cond = tp_cond and (cov_img >= T_cov)

            if tp_cond:
                label = "TP"
            else:
                # if cov_img < T_cov:
                #     label = "FN"
                # elif cov_pred < T_cov:
                #     label = "FP"
                # else:
                #     label = "FN"  # ambiguous policy
                if cov_img < T_match and cov_img < cov_pred:
                    label = "FN"
                elif cov_pred < T_match and cov_pred < cov_img:
                    label = "FP"
                else: # if we compare cov to T_match actually there will not be any ambiguous cases, but we keep the old policy for safety (e.g. if cov_pred == cov_img )
                    label = "FN"  # ambiguous -> FN


            # hist only when real metrics exist
            cldice_sum_all += float(cldice)
            cldice_count_all += 1
            b_c = min(hist_bins - 1, int(np.clip(cldice, 0, 0.999999) * hist_bins))
            cldice_hist_all[b_c] += 1

            b_ci = min(hist_bins - 1, int(np.clip(cov_img, 0, 0.999999) * hist_bins))
            b_cp = min(hist_bins - 1, int(np.clip(cov_pred, 0, 0.999999) * hist_bins))
            cov_img_hist[b_ci] += 1
            cov_pred_hist[b_cp] += 1

        # cache per-chunk metrics for later relabeling
        if store_chunk_metrics:
            z0_, z1_ = _slc_to_coords(slc[0])
            y0_, y1_ = _slc_to_coords(slc[1])
            x0_, x1_ = _slc_to_coords(slc[2])
            chunk_metrics.append({
                "z0": z0_, "z1": z1_, "y0": y0_, "y1": y1_, "x0": x0_, "x1": x1_,
                "has_img": has_img,
                "has_pred": has_pred,
                "cov_img": float(cov_img) if np.isfinite(cov_img) else np.nan,
                "cov_pred": float(cov_pred) if np.isfinite(cov_pred) else np.nan,
                "cldice": float(cldice) if np.isfinite(cldice) else np.nan,
                "GTvox": GTvox,
                "PRvox": PRvox,
                "GTsk": GTsk,
                "PRsk": PRsk,
            })

        # examples (subset)
        if keep_examples:
            prio = _example_priority(label, cov_img, cov_pred, cldice, T_match, T_cov) if example_strategy == "borderline" else 0.0
            ex = ChunkExample(
                slc=slc,
                has_img=has_img,
                has_pred=has_pred,
                label=label,
                cov_img=float(cov_img) if np.isfinite(cov_img) else np.nan,
                cov_pred=float(cov_pred) if np.isfinite(cov_pred) else np.nan,
                cldice=float(cldice) if np.isfinite(cldice) else np.nan,
                GTvox=GTvox,
                PRvox=PRvox,
                GTsk=GTsk,
                PRsk=PRsk,
            )
            _maybe_add_example(examples, label, ex, max_examples_per_class, example_strategy, prio)

            # EARLY STOP only in examples_only
            if mode == "examples_only":
                done = all(len(examples[k]) >= max_examples_per_class for k in examples_need)
                if done:
                    break

        if mode == "examples_only":
            continue

        # full eval aggregation
        voxel_conf = update_confusion(voxel_conf, gt_b, pr_b)
        chunk_presence_conf = update_confusion(
            chunk_presence_conf,
            np.array([gt_pos], dtype=bool),
            np.array([pr_pos], dtype=bool),
        )
        # The initial logic does not match how we assign TP/FP/FN/TN labels, so we update chunk_struct_conf separately based on the final label.
        if label == "TP":
            chunk_struct_conf = Confusion(chunk_struct_conf.tp + 1, chunk_struct_conf.fp,     chunk_struct_conf.fn,     chunk_struct_conf.tn)
        elif label == "FP":
            chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp + 1, chunk_struct_conf.fn,     chunk_struct_conf.tn)
        elif label == "FN":
            chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp,     chunk_struct_conf.fn + 1, chunk_struct_conf.tn)
        else:
            chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp,     chunk_struct_conf.fn,     chunk_struct_conf.tn + 1)


    f_gt.close()
    f_pr.close()

    out = {
        "thresholds": {"t_gt": int(t_gt), "t_pred": int(t_pred)},
        "chunk_zyx": chunk_zyx,
        "roi_low": int(roi_low),
        "chunk_alpha": float(chunk_alpha),
        "do_tolerant": bool(do_tolerant),
        "cldice_params": {
            "voxel_size_zyx": tuple(map(float, voxel_size_zyx)),
            "tau": float(tau),
            "T_match": float(T_match),
            "T_cov": float(T_cov),
            "require_cov_img_for_tp": bool(require_cov_img_for_tp),
            "use_skeleton_to_skeleton": bool(use_skeleton_to_skeleton),
            "mode": mode,
            "example_strategy": example_strategy,
        },
        "chunk_accounting": {
            "total_chunks": int(total_chunks),
            "used_chunks": int(used_chunks),
            "skipped_chunks": int(skipped_chunks),
        },
        "chunk_metrics": chunk_metrics,
    }

    # examples serialization
    if keep_examples:
        def _strip_and_sort(lst):
            lst_sorted = sorted(lst, key=lambda x: x[0])
            return [x[1] for x in lst_sorted]
        examples_clean = {k: _strip_and_sort(v) for k, v in examples.items()}

        def _coords(s): return (int(s.start), int(s.stop))
        out["examples"] = {
            k: [{
                "z": _coords(ex.slc[0]),
                "y": _coords(ex.slc[1]),
                "x": _coords(ex.slc[2]),
                "has_img": ex.has_img,
                "has_pred": ex.has_pred,
                "label": ex.label,
                "cov_img": ex.cov_img,
                "cov_pred": ex.cov_pred,
                "cldice": ex.cldice,
                "GTvox": ex.GTvox,
                "PRvox": ex.PRvox,
                "GTsk": ex.GTsk,
                "PRsk": ex.PRsk,
            } for ex in examples_clean[k]]
            for k in examples_clean.keys()
        }

    if mode == "examples_only":
        return out

    out.update({
        "voxel_confusion": {
            "tp": voxel_conf.tp, "fp": voxel_conf.fp, "fn": voxel_conf.fn, "tn": voxel_conf.tn,
            "precision": voxel_conf.precision, "recall": voxel_conf.recall, "f1": voxel_conf.f1,
        },
        "chunk_presence_confusion": {
            "tp": chunk_presence_conf.tp, "fp": chunk_presence_conf.fp,
            "fn": chunk_presence_conf.fn, "tn": chunk_presence_conf.tn,
            "precision": chunk_presence_conf.precision,
            "recall": chunk_presence_conf.recall,
            "f1": chunk_presence_conf.f1,
        },
        "chunk_structure_confusion": {
            "tp": chunk_struct_conf.tp, "fp": chunk_struct_conf.fp,
            "fn": chunk_struct_conf.fn, "tn": chunk_struct_conf.tn,
            "precision": chunk_struct_conf.precision,
            "recall": chunk_struct_conf.recall,
            "f1": chunk_struct_conf.f1,
        },
        "chunk_cldice": {
            "mean_cldice": (cldice_sum_all / cldice_count_all) if cldice_count_all > 0 else np.nan,
            "count": int(cldice_count_all),
            "hist_bins": int(hist_bins),
            "hist": cldice_hist_all.tolist(),
        },
        "chunk_cov_img": {"hist_bins": int(hist_bins), "hist": cov_img_hist.tolist()},
        "chunk_cov_pred": {"hist_bins": int(hist_bins), "hist": cov_pred_hist.tolist()},
    })
    return out


def surface_coverages_and_sfdice(
    gt_mask: np.ndarray,
    pr_mask: np.ndarray,
    voxel_size_zyx: Tuple[float, float, float] = (2.5, 0.7575, 0.7575),
    tau: float = 2.0,
    eps: float = 1e-8,
    connectivity: int = 1,
    return_surfaces: bool = True,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Surface Dice with tolerance (sfDice-like) for 3D binary masks.

    Concept:
      - Extract surfaces S_gt and S_pr (boundary voxels).
      - Compute directional coverages within tau (in physical units):
          cov_img  = fraction of S_gt within tau of S_pr
          cov_pred = fraction of S_pr within tau of S_gt
      - Combine by harmonic mean:
          sfdice = 2*cov_img*cov_pred / (cov_img + cov_pred + eps)

    Args:
      gt_mask, pr_mask:
        3D arrays; treated as boolean masks (nonzero=True).
      voxel_size_zyx:
        Physical voxel size (µm) in (Z, Y, X). Used for anisotropic distances.
      tau:
        Tolerance distance in same physical units as voxel_size_zyx (µm).
      eps:
        Small value to avoid division by zero.
      connectivity:
        Structuring element connectivity for erosion (1=6-neighborhood, 2=18, 3=26).
      return_surfaces:
        If True, include surface arrays in dbg dict (can be memory heavy).

    Returns:
      cov_img, cov_pred, sfdice, dbg

      dbg contains:
        - GTsurf, PRsurf (surface voxel counts)
        - tau, voxel_size_zyx
        - optional 'S_gt', 'S_pr' if return_surfaces=True
    """
    gt = gt_mask.astype(bool)
    pr = pr_mask.astype(bool)

    # ---- surface extraction: surface = mask - eroded(mask) ----
    # Choose connectivity-aware structuring element
    st = ndi.generate_binary_structure(rank=3, connectivity=connectivity)

    if gt.any():
        gt_er = ndi.binary_erosion(gt, structure=st, border_value=0)
        S_gt = gt & (~gt_er)
    else:
        S_gt = np.zeros_like(gt, dtype=bool)

    if pr.any():
        pr_er = ndi.binary_erosion(pr, structure=st, border_value=0)
        S_pr = pr & (~pr_er)
    else:
        S_pr = np.zeros_like(pr, dtype=bool)

    n_gt = int(S_gt.sum())
    n_pr = int(S_pr.sum())

    dbg: Dict[str, Any] = {
        "GTsurf": n_gt,
        "PRsurf": n_pr,
        "tau": float(tau),
        "voxel_size_zyx": tuple(map(float, voxel_size_zyx)),
        "connectivity": int(connectivity),
    }

    if return_surfaces:
        dbg["S_gt"] = S_gt
        dbg["S_pr"] = S_pr

    # ---- empty cases (match clDice-style conventions) ----
    # both empty: perfect match
    if n_gt == 0 and n_pr == 0:
        return 1.0, 1.0, 1.0, dbg
    # GT has no surface but PR does: PR is "extra"
    if n_gt == 0 and n_pr > 0:
        return 1.0, 0.0, 0.0, dbg
    # PR has no surface but GT does: PR misses everything
    if n_pr == 0 and n_gt > 0:
        return 0.0, 1.0, 0.0, dbg

    # ---- distance transforms in physical units ----
    # distance_transform_edt computes distance to nearest zero voxel,
    # so we invert the surface masks (~S_*), where surface voxels are False.
    d_to_PR = ndi.distance_transform_edt(~S_pr, sampling=voxel_size_zyx)
    d_to_GT = ndi.distance_transform_edt(~S_gt, sampling=voxel_size_zyx)

    # directional coverages
    cov_img = float((d_to_PR[S_gt] <= tau).mean())   # GT surface covered by PR surface
    cov_pred = float((d_to_GT[S_pr] <= tau).mean())  # PR surface covered by GT surface

    sfdice = float((2.0 * cov_img * cov_pred) / (cov_img + cov_pred + eps))

    return cov_img, cov_pred, sfdice, dbg


def chunk_based_eval_sfdice(
    ims_path: str,
    level: int,
    ch_gt: int,
    ch_pred: int,
    t_gt: int,
    t_pred: int,
    chunk_zyx: Tuple[int, int, int] = (20, 64, 64),
    roi_low: int = 1,
    chunk_alpha: float = 0.01,
    do_tolerant: bool = False,

    # sfDice params
    voxel_size_zyx: Tuple[float, float, float] = (2.5, 0.7575, 0.7575),
    tau: float = 2.0,
    T_match: float = 0.7,
    T_cov: float = 0.5,
    require_cov_img_for_tp: bool = True,

    hist_bins: int = 20,

    # caching: store per-chunk metrics for later threshold sweeps
    store_chunk_metrics: bool = True,
) -> Dict:
    """
    Chunk-based evaluation using sfDice (surface Dice with tolerance tau).

    Labeling logic (same policy as clDice version):
      - If not (has_img and has_pred): TN/FN/FP by presence.
      - Else:
          compute (cov_img, cov_pred, sfdice)
          TP if (sfdice >= T_match) and (optionally cov_img >= T_cov)
          else FN/FP based on which coverage fails, else FN (ambiguous)

    Returns a dict similar to chunk_based_eval_cldice, but with sfdice fields.
    """

    f_gt, ds_gt = open_ims_dataset(ims_path, level, ch_gt)
    f_pr, ds_pr = open_ims_dataset(ims_path, level, ch_pred)
    shape = ds_gt.shape
    assert ds_pr.shape == shape, "GT and Pred channels must have the same shape."

    voxel_conf = Confusion(0, 0, 0, 0)
    chunk_presence_conf = Confusion(0, 0, 0, 0)
    chunk_struct_conf = Confusion(0, 0, 0, 0)

    sfdice_hist_all = np.zeros(hist_bins, dtype=np.int64)
    sfdice_sum_all = 0.0
    sfdice_count_all = 0

    cov_img_hist = np.zeros(hist_bins, dtype=np.int64)
    cov_pred_hist = np.zeros(hist_bins, dtype=np.int64)

    total_chunks = 0
    used_chunks = 0
    skipped_chunks = 0

    halo = 1 if do_tolerant else 0
    chunk_metrics: List[dict] = []

    def _slc_to_coords(s: slice):
        return int(s.start), int(s.stop)

    try:
        for slc in iter_chunks_zyx(shape, chunk_zyx):
            total_chunks += 1

            zsl, ysl, xsl = slc
            z0 = max(0, zsl.start - halo); z1 = min(shape[0], zsl.stop + halo)
            y0 = max(0, ysl.start - halo); y1 = min(shape[1], ysl.stop + halo)
            x0 = max(0, xsl.start - halo); x1 = min(shape[2], xsl.stop + halo)
            slc_h = (slice(z0, z1), slice(y0, y1), slice(x0, x1))

            gt_h = read_block(ds_gt, slc_h).astype(np.uint8)
            pr_h = read_block(ds_pr, slc_h).astype(np.uint8)

            roi_h = (gt_h >= roi_low) | (pr_h >= roi_low)
            if roi_h.sum() == 0:
                skipped_chunks += 1
                continue
            used_chunks += 1

            gt_b_h = (gt_h >= t_gt) & roi_h
            pr_b_h = (pr_h >= t_pred) & roi_h

            # NOTE: This dilates GT only -> bias. Consider disabling or also dilating PR.
            if do_tolerant:
                gt_b_h = dilate_3d_radius1(gt_b_h)

            iz0 = zsl.start - z0; iz1 = iz0 + (zsl.stop - zsl.start)
            iy0 = ysl.start - y0; iy1 = iy0 + (ysl.stop - ysl.start)
            ix0 = xsl.start - x0; ix1 = ix0 + (xsl.stop - xsl.start)

            gt_b = gt_b_h[iz0:iz1, iy0:iy1, ix0:ix1]
            pr_b = pr_b_h[iz0:iz1, iy0:iy1, ix0:ix1]

            GTvox = int(gt_b.sum())
            PRvox = int(pr_b.sum())

            # presence (alpha fraction)
            gt_pos, pr_pos = compute_chunk_label_confusion(gt_b, pr_b, alpha=chunk_alpha)
            has_img, has_pred = bool(gt_pos), bool(pr_pos)

            # defaults
            cov_img = cov_pred = sfdice = np.nan
            GTsurf = 0 if GTvox == 0 else -1
            PRsurf = 0 if PRvox == 0 else -1

            # presence-only labeling if not both present
            # if not (has_img and has_pred):
            if (not has_img) and (not has_pred):
                label = "TN"
                # elif has_img and (not has_pred):
                #     label = "FN"
                # else:  # (not has_img) and has_pred
                #     label = "FP"
            else:
                cov_img, cov_pred, sfdice, dbg = surface_coverages_and_sfdice(
                    gt_mask=gt_b,
                    pr_mask=pr_b,
                    voxel_size_zyx=voxel_size_zyx,
                    tau=tau,
                )

                if isinstance(dbg, dict):
                    GTsurf = int(dbg.get("GTsurf", GTsurf))
                    PRsurf = int(dbg.get("PRsurf", PRsurf))

                tp_cond = (sfdice >= T_match)
                if require_cov_img_for_tp:
                    tp_cond = tp_cond and (cov_img >= T_cov)

                if tp_cond:
                    label = "TP"
                else:
                    # same policy as your clDice code
                    # if cov_img < T_cov:
                    #     label = "FN"
                    # elif cov_pred < T_cov:
                    #     label = "FP"
                    # else:
                    #     label = "FN"  # ambiguous -> FN
                    # Check which coverage is worse to assign FP vs FN, else default to FN for ambiguity
                    if cov_img < T_match and cov_img < cov_pred:
                        label = "FN"
                    elif cov_pred < T_match and cov_pred < cov_img:
                        label = "FP"
                    else: # if we compare cov to T_match actually there will not be any ambiguous cases, but we keep the old policy for safety (e.g. if cov_pred == cov_img )
                        label = "FN"  # ambiguous -> FN

                # hist/mean only when metrics exist
                sfdice_sum_all += float(sfdice)
                sfdice_count_all += 1

                b_s = min(hist_bins - 1, int(np.clip(sfdice, 0, 0.999999) * hist_bins))
                sfdice_hist_all[b_s] += 1

                b_ci = min(hist_bins - 1, int(np.clip(cov_img, 0, 0.999999) * hist_bins))
                b_cp = min(hist_bins - 1, int(np.clip(cov_pred, 0, 0.999999) * hist_bins))
                cov_img_hist[b_ci] += 1
                cov_pred_hist[b_cp] += 1

            # cache per-chunk metrics
            if store_chunk_metrics:
                z0_, z1_ = _slc_to_coords(slc[0])
                y0_, y1_ = _slc_to_coords(slc[1])
                x0_, x1_ = _slc_to_coords(slc[2])
                chunk_metrics.append({
                    "z0": z0_, "z1": z1_, "y0": y0_, "y1": y1_, "x0": x0_, "x1": x1_,
                    "has_img": has_img,
                    "has_pred": has_pred,
                    "label": label,
                    "cov_img": float(cov_img) if np.isfinite(cov_img) else np.nan,
                    "cov_pred": float(cov_pred) if np.isfinite(cov_pred) else np.nan,
                    "sfdice": float(sfdice) if np.isfinite(sfdice) else np.nan,
                    "GTvox": GTvox,
                    "PRvox": PRvox,
                    "GTsurf": GTsurf,
                    "PRsurf": PRsurf,
                })

            # full eval aggregation
            voxel_conf = update_confusion(voxel_conf, gt_b, pr_b)
            chunk_presence_conf = update_confusion(
                chunk_presence_conf,
                np.array([gt_pos], dtype=bool),
                np.array([pr_pos], dtype=bool),
            )

            # structure confusion based on final label
            if label == "TP":
                chunk_struct_conf = Confusion(chunk_struct_conf.tp + 1, chunk_struct_conf.fp,     chunk_struct_conf.fn,     chunk_struct_conf.tn)
            elif label == "FP":
                chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp + 1, chunk_struct_conf.fn,     chunk_struct_conf.tn)
            elif label == "FN":
                chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp,     chunk_struct_conf.fn + 1, chunk_struct_conf.tn)
            else:
                chunk_struct_conf = Confusion(chunk_struct_conf.tp,     chunk_struct_conf.fp,     chunk_struct_conf.fn,     chunk_struct_conf.tn + 1)

    finally:
        f_gt.close()
        f_pr.close()

    out = {
        "level": int(level),
        "channels": {"ch_gt": int(ch_gt), "ch_pred": int(ch_pred)},
        "thresholds": {"t_gt": int(t_gt), "t_pred": int(t_pred)},
        "chunk_zyx": tuple(map(int, chunk_zyx)),
        "roi_low": int(roi_low),
        "chunk_alpha": float(chunk_alpha),
        "do_tolerant": bool(do_tolerant),
        "sfdice_params": {
            "voxel_size_zyx": tuple(map(float, voxel_size_zyx)),
            "tau": float(tau),
            "T_match": float(T_match),
            "T_cov": float(T_cov),
            "require_cov_img_for_tp": bool(require_cov_img_for_tp),
        },
        "chunk_accounting": {
            "total_chunks": int(total_chunks),
            "used_chunks": int(used_chunks),
            "skipped_chunks": int(skipped_chunks),
        },
        "chunk_metrics": chunk_metrics,
        "voxel_confusion": {
            "tp": voxel_conf.tp, "fp": voxel_conf.fp, "fn": voxel_conf.fn, "tn": voxel_conf.tn,
            "precision": voxel_conf.precision, "recall": voxel_conf.recall, "f1": voxel_conf.f1,
        },
        "chunk_presence_confusion": {
            "tp": chunk_presence_conf.tp, "fp": chunk_presence_conf.fp,
            "fn": chunk_presence_conf.fn, "tn": chunk_presence_conf.tn,
            "precision": chunk_presence_conf.precision,
            "recall": chunk_presence_conf.recall,
            "f1": chunk_presence_conf.f1,
        },
        "chunk_structure_confusion": {
            "tp": chunk_struct_conf.tp, "fp": chunk_struct_conf.fp,
            "fn": chunk_struct_conf.fn, "tn": chunk_struct_conf.tn,
            "precision": chunk_struct_conf.precision,
            "recall": chunk_struct_conf.recall,
            "f1": chunk_struct_conf.f1,
        },
        "chunk_sfdice": {
            "mean_sfdice": (sfdice_sum_all / sfdice_count_all) if sfdice_count_all > 0 else np.nan,
            "count": int(sfdice_count_all),
            "hist_bins": int(hist_bins),
            "hist": sfdice_hist_all.tolist(),
        },
        "chunk_cov_img": {"hist_bins": int(hist_bins), "hist": cov_img_hist.tolist()},
        "chunk_cov_pred": {"hist_bins": int(hist_bins), "hist": cov_pred_hist.tolist()},
    }

    return out