from sklearn import metrics
from skimage import measure

import cv2
import numpy as np
import pandas as pd
import auproUtil


def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = metrics.precision_recall_curve(anomaly_ground_truth_labels, anomaly_prediction_weights)
    f1_scores = 2 * (precision * recall) / (precision + recall)

    best_threshold = thresholds[np.argmax(f1_scores)]
    best_precision = precision[np.argmax(f1_scores)]
    best_recall = recall[np.argmax(f1_scores)]
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).
    """
    auroc = metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0. if path == 'training' else metrics.average_precision_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel()

    auroc = metrics.roc_auc_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)
    ap = 0. if path == 'training' else metrics.average_precision_score(flat_ground_truth_masks.astype(int), flat_anomaly_segmentations)

    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    """
    Compute PRO curve and its integral using auproUtil.

    Args:
        masks: List of ground truth masks or 3D numpy array
        amaps: List of anomaly maps or 3D numpy array
        num_th: Number of thresholds (not used by auproUtil, kept for compatibility)

    Returns:
        pro_auc: Area under the PRO curve integrated up to 30% FPR
    """
    # Convert inputs to list format if they are arrays
    if isinstance(masks, np.ndarray):
        if masks.ndim == 3:
            masks = [masks[i] for i in range(masks.shape[0])]
        else:
            masks = [masks]

    if isinstance(amaps, np.ndarray):
        if amaps.ndim == 3:
            amaps = [amaps[i] for i in range(amaps.shape[0])]
        else:
            amaps = [amaps]

    # Calculate AU-PRO using auproUtil with 30% FPR integration limit
    au_pro, _ = auproUtil.calculate_au_pro(gts=masks, predictions=amaps,
                                           integration_limit=0.3, num_thresholds=100)

    return au_pro
