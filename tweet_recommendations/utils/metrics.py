from typing import Any, List

import numpy as np


def get_recall_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    if not ground_truth:
        return 1
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(ground_truth) & set(predicted)) / len(ground_truth)


def get_mean_recall_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    return float(
        np.mean([get_recall_at_k(a, p, k) for a, p in zip(ground_truth, predicted)])
    )


def get_precision_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    if not ground_truth:
        return 1
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(ground_truth) & set(predicted)) / len(predicted)


def get_mean_precision_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    return float(
        np.mean([get_precision_at_k(a, p, k) for a, p in zip(ground_truth, predicted)])
    )


def get_f1_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    if not ground_truth:
        return 1
    precision = get_precision_at_k(ground_truth, predicted, k)
    recall = get_recall_at_k(ground_truth, predicted, k)
    if precision == 0 and recall == 0:
        return 0
    return 2*precision*recall / (precision + recall)


def get_mean_f1_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    return float(
        np.mean([get_f1_at_k(a, p, k) for a, p in zip(ground_truth, predicted)])
    )
