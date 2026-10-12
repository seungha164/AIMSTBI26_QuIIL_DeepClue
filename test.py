import argparse
import os
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.metrics import HausdorffDistanceMetric, SurfaceDistanceMetric
from tqdm import tqdm

# 2D visualization settings
VIS_MAX_SLICES = 150
VIS_SLICE_STEP = 2


def load_mask(nifti_path):
    img = nib.load(str(nifti_path))
    data = np.asanyarray(img.dataobj)
    return img, data


def compute_case_metrics(pred_mask, gt_mask):
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    pred_sum = int(pred_bin.sum())
    gt_sum = int(gt_bin.sum())

    inter = int(np.logical_and(pred_bin, gt_bin).sum())
    denom = pred_sum + gt_sum
    dice = 1.0 if denom == 0 else (2.0 * inter / denom)

    # Surface distances are undefined when either mask is empty.
    if pred_sum == 0 or gt_sum == 0:
        return dice, float("nan"), float("nan")

    # Metric objects are created locally so each worker process owns its own state.
    hd_metric = HausdorffDistanceMetric(include_background=True, percentile=95, reduction="mean")
    assd_metric = SurfaceDistanceMetric(include_background=True, symmetric=True, reduction="mean")

    pred_tensor = torch.from_numpy(pred_bin[None, None].astype(np.float32))
    gt_tensor = torch.from_numpy(gt_bin[None, None].astype(np.float32))

    hd_metric(y_pred=pred_tensor, y=gt_tensor)
    assd_metric(y_pred=pred_tensor, y=gt_tensor)

    return dice, hd_metric.aggregate().item(), assd_metric.aggregate().item()


def normalize_volume_for_vis(img_vol, p_low=1, p_high=99):
    """Percentile-clip a volume and rescale it to uint8 for display."""
    vol = img_vol.astype(np.float32)

    nz = vol[vol > 0]
    ref = nz if nz.size > 0 else vol
    lo = np.percentile(ref, p_low)
    hi = np.percentile(ref, p_high)

    vol = np.clip(vol, lo, hi)
    vol = (vol - lo) / (hi - lo + 1e-8)
    return (vol * 255).astype(np.uint8)


def overlay_mask(img_rgb, mask, cv2):
    if not mask.any():
        return img_rgb.copy()

    colormap = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_rgb, 0.4, colormap, 0.6, 0)

    out = img_rgb.copy()
    out[mask > 0] = blended[mask > 0]
    return out


def save_slice_visualization(img_u8, gt_slice, pred_slice, save_path):
    """Write an [image | GT overlay | prediction overlay] triptych."""
    import cv2

    img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
    combined = np.concatenate(
        [img_rgb, overlay_mask(img_rgb, gt_slice, cv2), overlay_mask(img_rgb, pred_slice, cv2)],
        axis=1,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, combined)


def visualize_case(subject_id, gt_path, pred_path, gt_data, pred_data):
    from concurrent.futures import ThreadPoolExecutor

    img_path = str(gt_path).replace("labelsTr", "imagesTr").replace(".nii.gz", "_0000.nii.gz")
    if not os.path.exists(img_path):
        return

    img_vol_u8 = normalize_volume_for_vis(np.asanyarray(nib.load(img_path).dataobj))
    if img_vol_u8.shape != gt_data.shape:
        return

    save_root = os.path.join(os.path.dirname(str(pred_path)), "vis2D")
    depth = min(VIS_MAX_SLICES, img_vol_u8.shape[2])

    jobs = [
        (
            img_vol_u8[:, :, idx],
            (gt_data[:, :, idx] > 0).astype(np.uint8),
            (pred_data[:, :, idx] > 0).astype(np.uint8),
            os.path.join(save_root, f"{subject_id}_slice{idx:03d}.png"),
        )
        for idx in range(0, depth, VIS_SLICE_STEP)
    ]

    if jobs:
        with ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(lambda job: save_slice_visualization(*job), jobs))


def evaluate_one_case(task):
    pred_path, gt_dir, vis_2d = task

    subject_id = pred_path.name.removesuffix(".nii.gz")
    gt_path = gt_dir / pred_path.name

    if not gt_path.exists():
        return {"Subject_ID": subject_id, "Error": f"GT not found: {gt_path}"}

    try:
        _, gt_data = load_mask(gt_path)
        _, pred_data = load_mask(pred_path)

        if gt_data.shape != pred_data.shape:
            return {
                "Subject_ID": subject_id,
                "Error": f"Shape mismatch: GT {gt_data.shape}, Pred {pred_data.shape}",
            }

        dice, hd95, assd = compute_case_metrics(pred_data, gt_data)

        if vis_2d:
            visualize_case(subject_id, gt_path, pred_path, gt_data, pred_data)

        return {
            "Subject_ID": subject_id,
            "Dice_Score": dice,
            "ASSD": assd,
            "HD95": hd95,
            "Error": "",
        }

    except Exception as exc:
        return {"Subject_ID": subject_id, "Error": repr(exc)}


def evaluate_folder(gt_dir, pred_dir, eda_csv, num_workers=8, vis_2d=False):
    pred_paths = sorted(pred_dir.glob("*.nii.gz"))
    print(f"Predictions found: {len(pred_paths)}")

    # Keep only lesion-positive subjects; lesion-free cases are scored separately.
    if eda_csv.exists():
        eda_df = pd.read_csv(eda_csv)
        lesion_subjects = set(eda_df[eda_df["is_lesion"]]["case_id"].astype(str))
        pred_paths = [p for p in pred_paths if p.name.removesuffix(".nii.gz") in lesion_subjects]
        print(f"Predictions after lesion-only filter: {len(pred_paths)}")
    else:
        print(f"EDA file not found, skipping lesion-only filter: {eda_csv}")

    if not pred_paths:
        return pd.DataFrame([])

    tasks = [(p, gt_dir, vis_2d) for p in pred_paths]

    if num_workers <= 1:
        results = [evaluate_one_case(task) for task in tqdm(tasks)]
    else:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap_unordered(evaluate_one_case, tasks), total=len(tasks)))

    df = pd.DataFrame(results)

    if "Error" in df.columns:
        error_df = df[df["Error"].fillna("") != ""]
        if len(error_df) > 0:
            print("\n[Failed cases]")
            print(error_df[["Subject_ID", "Error"]].to_string(index=False))
        df = df[df["Error"].fillna("") == ""].drop(columns=["Error"])

    return df


def print_summary(df_scores):
    print("\n[Per-subject scores]")
    print(df_scores.to_string(index=False))

    print("\n[Overall summary]")
    print(f"Subjects evaluated: {len(df_scores)}")
    for column in ["Dice_Score", "HD95", "ASSD"]:
        values = df_scores[column]
        print(f"Mean {column}: {values.mean():.4f} (+/- {values.std():.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_dir",
        "--prediction_dir",
        dest="pred_dir",
        type=str,
        default="nnUNet_results/Dataset500_AIMSTBI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/test_best",
    )
    parser.add_argument("--gt_dir", type=str, default="DATA/nnUNet_raw/Dataset500_AIMSTBI/labelsTr")
    parser.add_argument("--eda_csv", type=str, default="analysis/ALL_CASE_EDA.csv")
    parser.add_argument("--num_workers", type=int, default=min(cpu_count(), 8))
    parser.add_argument("--output_csv", type=str, default="Task2_segmentation_result.csv")
    parser.add_argument("--vis2d", action="store_true", help="Save 2D overlay images per slice")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        print(f"Prediction directory does not exist: {pred_dir}")
        return

    output_csv = Path(args.output_csv)
    if not output_csv.is_absolute() and output_csv.parent == Path("."):
        output_csv = pred_dir.parent / output_csv

    print("Starting segmentation evaluation...")
    print(f"GT dir:   {gt_dir}")
    print(f"Pred dir: {pred_dir}")
    print(f"Workers:  {args.num_workers}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df_scores = evaluate_folder(
            gt_dir,
            pred_dir,
            Path(args.eda_csv),
            num_workers=args.num_workers,
            vis_2d=args.vis2d,
        )

    if df_scores.empty:
        print("No cases to evaluate. Please check the directory paths.")
        return

    df_scores = df_scores.sort_values("Subject_ID").reset_index(drop=True)
    print_summary(df_scores)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    main()