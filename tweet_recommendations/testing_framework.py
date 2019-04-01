import random

import numpy as np
import pandas as pd
import os
import pickle

from tweet_recommendations.utils.metrics import (
    get_rank_dcg_at_k,
    get_recall_at_k,
    get_average_precision_at_k,
)

random.seed(1337)
config = {"K": 3}


def get_k_random_hashtags_from_hashtag_data(hashtag_data: pd.DataFrame, k=10):
    hashtag_range = len(hashtag_data)
    hashtag_list = []
    for i in range(k):
        hashtag_list.append(
            hashtag_data.iloc[random.randint(0, hashtag_range - 1)]["hashtag"]
        )
    return hashtag_list


def get_hashtag_rank_for_given_tweets(tweet: str, hashtag_data: pd.DataFrame, config: dict):
    return get_k_random_hashtags_from_hashtag_data(hashtag_data, config["K"])


def get_map_value_for_tweets(
        test_set: pd.DataFrame,
        config: dict,
        hashtag_data=None,
        recommendation_function=get_hashtag_rank_for_given_tweets,
):
    test_recall, test_mAP, test_ndcg = zip(
        *test_set.apply(
            lambda row: get_metrices_for_given_dataset(
                config, hashtag_data, row, recommendation_function
            ),
            axis=1
        ).tolist()
    )
    return (
        np.mean(np.asarray(test_recall)),
        np.mean(np.asarray(test_mAP)),
        np.mean(np.asarray(test_ndcg)),
    )


def get_metrices_for_given_dataset(config, hashtag_data, row, recommendation_function):
    predicted_hashtag_rank = recommendation_function(row["text"], hashtag_data, config)
    expected_hashtags = row["expected_hashtags"]
    return (
        get_recall_at_k(expected_hashtags, predicted_hashtag_rank, config["K"]),
        get_average_precision_at_k(
            expected_hashtags, predicted_hashtag_rank, config["K"]
        ),
        get_rank_dcg_at_k(expected_hashtags, predicted_hashtag_rank, config["K"]),
    )