__author__ = "Ezequiel de la Rosa"

import numpy as np
import warnings
from panoptica import (
    InputType,
    Panoptica_Evaluator,
    ConnectedComponentsInstanceApproximator,
    NaiveThresholdMatching,
)
from sklearn.metrics import average_precision_score

def compute_absolute_volume_difference(im1, im2, voxel_size):
    """
    Computes the absolute volume difference between two masks.

    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size as 'ground_truth'. If not boolean, it will be converted.
    voxel_size : scalar, float (ml)
        If not float, it will be converted.

    Returns
    -------
    abs_vol_diff : float, measured in ml.
        Absolute volume difference as a float.
        Maximum similarity = 0
        No similarity = inf


    Notes
    -----
    The order of inputs is irrelevant. The result will be identical if `im1` and `im2` are switched.
    """

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = voxel_size.astype(float)

    if im1.shape != im2.shape:
        warnings.warn(
            "Shape mismatch: ground_truth and prediction have difference shapes."
            " The absolute volume difference is computed with mismatching shape masks"
        )

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size
    abs_vol_diff = np.abs(ground_truth_volume - prediction_volume)

    return abs_vol_diff


def compute_dice_f1_instance_difference(ground_truth, prediction, empty_value=1.0, return_counts=False):
    """
    Computes the lesion-wise F1-score, instance count difference, and Dice score between two masks.

    Parameters
    ----------
    ground_truth : array-like, int
        Any array of arbitrary size. If not int, it will be converted.
    prediction: array-like, bool
        Any other array of identical size as 'ground_truth'. If not int, it will be converted.
    empty_value : scalar, float.
    return_counts : bool, optional
        If True, also return predicted and ground truth instance counts.
    
    Returns
    -------
    f1_score : float
        Instance-wise F1-score as float.
        Max score = 1
        Min score = 0
        If both images are empty F1-Score = empty_value

    -------
    dice_score : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1
        No similarity = 0
        If both images are empty (sum equal to zero) = empty_value
    -------
    instance_count_difference : int
        Absolute instance count difference as integer.
        Maximum similarity = 0
        No similarity = --> inf
        
    """
    
    ground_truth = np.asarray(ground_truth).astype(int)
    prediction = np.asarray(prediction).astype(int)

    evaluator = Panoptica_Evaluator(
        expected_input=InputType.SEMANTIC,
        instance_approximator=ConnectedComponentsInstanceApproximator(),
        instance_matcher=NaiveThresholdMatching(matching_threshold=0.2),
    )
    
    result = evaluator.evaluate(prediction, ground_truth, verbose=False)["ungrouped"]
    
    instance_count_difference = abs(result.n_ref_instances - result.n_pred_instances) # compute lesion count difference
    
    if result.n_ref_instances== 0 and result.n_pred_instances==0:
        f1_score = empty_value
        dice_score = empty_value
    else:
        f1_score = result.rq # get f1-score        
        dice_score = result.global_bin_dsc
    
    if return_counts:
        return f1_score, instance_count_difference, dice_score, result.n_pred_instances, result.n_ref_instances
    return f1_score, instance_count_difference, dice_score

#* 기존 코드
# def compute_pr_auc_average_precision(ground_truth, lesion_probability):
#     """
#     Computes PR-AUC as Average Precision (AP) over voxel-wise soft probabilities.

#     ground_truth: binary GT mask, shape [X, Y, Z]
#     lesion_probability: lesion probability map, same shape as GT, values [0, 1]

#     Returns
#     -------
#     ap : float
#         Average Precision. If GT is empty, returns np.nan.
#     """

#     gt = np.asarray(ground_truth).astype(bool).ravel()
#     prob = np.asarray(lesion_probability).astype(np.float32).ravel()

#     if gt.shape != prob.shape:
#         raise ValueError(f"Shape mismatch: GT {gt.shape}, probability {prob.shape}")

#     # Challenge definition: AP undefined for empty GT
#     if gt.sum() == 0:
#         return np.nan

#     prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
#     prob = np.clip(prob, 0.0, 1.0)

#     return average_precision_score(gt.astype(np.uint8), prob)

# * NEW: bg 제외
def compute_pr_auc_average_precision(ground_truth, lesion_probability):
    gt = np.asarray(ground_truth).astype(bool).ravel()
    prob = np.asarray(lesion_probability).astype(np.float32).ravel()

    if gt.shape != prob.shape:
        raise ValueError(f"Shape mismatch: GT {gt.shape}, probability {prob.shape}")

    if gt.sum() == 0:
        return np.nan

    prob = np.nan_to_num(prob, nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)

    # 💡 고속화 핵심: 둘 다 0인 무의미한 배경 픽셀 대거 필터링 (결과 완벽 동일)
    valid_mask = (gt > 0) | (prob > 0.0)
    gt = gt[valid_mask]
    prob = prob[valid_mask]

    return average_precision_score(gt.astype(np.uint8), prob)
