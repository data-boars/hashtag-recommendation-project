from typing import Any, List

import numpy as np


def get_average_precision_at_k(
    ground_truth: List[Any], predicted: List[Any], k=5
) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in ground_truth and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not ground_truth:
        return 0.0

    return score / min(len(ground_truth), k)


def get_mean_average_precision_at_k(
    ground_truth: List[Any], predicted: List[Any], k=5
) -> float:
    return float(
        np.mean(
            [
                get_average_precision_at_k(a, p, k)
                for a, p in zip(ground_truth, predicted)
            ]
        )
    )


def get_recall_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
    return len(set(ground_truth) & set(predicted)) / len(ground_truth)


def get_mean_recall_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    return float(
        np.mean([get_recall_at_k(a, p, k) for a, p in zip(ground_truth, predicted)])
    )


def _order_lists(ground_truth: List[Any], predicted: List[Any]):
    pair_ref_list = sorted([x for x in enumerate(ground_truth)], key=lambda x: x[1])
    mapped_hyp_list = [predicted[x[0]] for x in pair_ref_list]

    return [x[1] for x in pair_ref_list], mapped_hyp_list


def get_rank_dcg_at_k(ground_truth: List[Any], predicted: List[Any], k=5) -> float:
    reference_list, hypothesis_list = _order_lists(ground_truth[:k], predicted[:k])

    ordered_list = reference_list[:]
    ordered_list.sort(reverse=True)

    high_rank = float(len(set(reference_list)))
    reverse_rank = 1.0
    relative_rank_list = [high_rank]
    reverse_rank_list = [reverse_rank]

    for index, rank in enumerate(ordered_list[:-1]):
        if ordered_list[index + 1] != rank:
            high_rank -= 1.0
            reverse_rank += 1.0
        relative_rank_list.append(high_rank)
        reverse_rank_list.append(reverse_rank)

    reference_pair_list = [x for x in enumerate(reference_list)]
    sorted_reference_pairs = sorted(
        reference_pair_list, key=lambda p: p[1], reverse=True
    )
    rel_rank_reference_list = [0] * len(reference_list)
    for position, rel_rank in enumerate(relative_rank_list):
        rel_rank_reference_list[sorted_reference_pairs[position][0]] = rel_rank

    max_score = sum(
        [
            rank / reverse_rank_list[index]
            for index, rank in enumerate(relative_rank_list)
        ]
    )
    min_score = sum(
        [
            rank / reverse_rank_list[index]
            for index, rank in enumerate(reversed(relative_rank_list))
        ]
    )

    hypothesis_pair_list = [x for x in enumerate(hypothesis_list)]
    sorted_hypothesis_pairs = sorted(
        hypothesis_pair_list, key=lambda p: p[1], reverse=True
    )
    eval_score = sum(
        [
            rel_rank_reference_list[pair[0]] / reverse_rank_list[index]
            for index, pair in enumerate(sorted_hypothesis_pairs)
        ]
    )
    if max_score - min_score == 0:
        return 0.0

    return (eval_score - min_score) / (max_score - min_score)
