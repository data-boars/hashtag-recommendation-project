import random

import pandas as pd
import numpy as np

from tweet_recommendations.utils.metrics import get_mean_average_precision_at_k, get_rank_dcg_at_k

random.seed(1337)

use_w2v_embedding = True

data = pd.DataFrame(data={"hashtag": ["a", "b", "c", "d", "e", "f", "g", "h"]})
config = {"hashtag_count": 3}
testdict = {
    "tweet": ["123", "456", "789"],
    "expected_hashtags": [["a", "c", "g"], ["c", "d", "f"], ["a", "d", "h"]],
}
testset = pd.DataFrame(data=testdict)


def get_k_random_hashtags_from_hashtag_data(hashtag_data, k=10):
    hashtag_range = len(hashtag_data)
    hashtag_list = []
    for i in range(k):
        hashtag_list.append(
            hashtag_data.iloc[random.randint(0, hashtag_range - 1)]["hashtag"]
        )
    return hashtag_list


def get_hashtag_rank_for_given_tweets(tweet, hashtag_data: pd.DataFrame, config: dict):
    return get_k_random_hashtags_from_hashtag_data(
        hashtag_data, config["hashtag_count"]
    )


def get_map_value_for_tweet(test_set, hashtag_data=None):
    test_mAP = []
    test_ndcg = []
    for index, row in test_set.iterrows():
        predicted_hashtag_rank = get_hashtag_rank_for_given_tweets(
            row["tweet"], hashtag_data, config
        )
        expected_hashtags = row["expected_hashtags"]
        test_mAP.append(
            get_mean_average_precision_at_k(expected_hashtags, predicted_hashtag_rank, 3)
        )
        test_ndcg.append(get_rank_dcg_at_k(expected_hashtags, predicted_hashtag_rank, 3))
    return np.mean(np.asarray(test_mAP)), np.mean(np.asarray(test_ndcg))


print(get_map_value_for_tweet(testset, data))
