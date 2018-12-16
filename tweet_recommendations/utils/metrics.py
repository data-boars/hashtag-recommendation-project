from typing import Any, List

import numpy as np


def average_precision_at_k(actual: List[Any], predicted: List[Any], k=5) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mean_average_precision_at_k(actual: List[Any], predicted: List[Any], k=5) -> float:
    return float(
        np.mean([average_precision_at_k(a, p, k) for a, p in zip(actual, predicted)])
    )
