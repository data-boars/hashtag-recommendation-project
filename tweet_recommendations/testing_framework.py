import random

import pandas as pd
import numpy as np

from tweet_recommendations.utils.metrics import (
    get_mean_average_precision_at_k,
    get_rank_dcg_at_k,
)

random.seed(1337)
config = {"rank_length": 3}


def get_k_random_hashtags_from_hashtag_data(hashtag_data: pd.DataFrame, k=10):
    hashtag_range = len(hashtag_data)
    hashtag_list = []
    for i in range(k):
        hashtag_list.append(
            hashtag_data.iloc[random.randint(0, hashtag_range - 1)]["hashtag"]
        )
    return hashtag_list


def get_hashtag_rank_for_given_tweets(
    tweet: str, hashtag_data: pd.DataFrame, config: dict
):
    return get_k_random_hashtags_from_hashtag_data(hashtag_data, config["rank_length"])


def get_map_value_for_tweet(test_set: pd.DataFrame, config: dict, hashtag_data=None):
    test_mAP, test_ndcg = zip(
        *test_set.apply(
            lambda row: get_metrices_for_given_dataset(config, hashtag_data, row),
            axis=1,
        ).tolist()
    )
    return np.mean(np.asarray(test_mAP)), np.mean(np.asarray(test_ndcg))


def get_metrices_for_given_dataset(config, hashtag_data, row):
    predicted_hashtag_rank = get_hashtag_rank_for_given_tweets(
        row["tweet"], hashtag_data, config
    )
    expected_hashtags = row["expected_hashtags"]
    return (
        get_mean_average_precision_at_k(
            expected_hashtags, predicted_hashtag_rank, config["rank_length"]
        ),
        get_rank_dcg_at_k(
            expected_hashtags, predicted_hashtag_rank, config["rank_length"]
        ),
    )
