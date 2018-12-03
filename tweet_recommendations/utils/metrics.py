import numpy as np


def average_precision_at_k(actual, predicted, k=5):
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


def mean_average_precision_at_k(actual, predicted, k=5):
    return np.mean([average_precision_at_k(a, p, k) for a, p in zip(actual, predicted)])
