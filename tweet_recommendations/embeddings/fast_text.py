from typing import List, Tuple

import numpy as np
import pandas as pd

import fastText
from tweet_recommendations.utils.constants import FASTTEXT_MODEL_PATH


def get_fasttext_tweets_embeddings(
    twitter_data: List[Tuple(str, List[str], str)]
) -> pd.DataFrame:
    id_list = []
    meaned_word_vector_list = []
    for tweet_id, tweet_word_list, status in twitter_data:
        id_list.append(tweet_id)
        meaned_word_vector_list.append(get_fasttext_tweet_embedding(tweet_word_list))
    return pd.DataFrame(
        data={"tweet_id": id_list, "tweet_embedding": meaned_word_vector_list}
    )


def get_fasttext_tweet_embedding(tweet_word_list: List[str]) -> np.ndarray:
    all_embeddings = []
    embedding_model = fastText.load_model(FASTTEXT_MODEL_PATH)
    for word in tweet_word_list:
        all_embeddings.append(embedding_model.get_word_vector(word))
    return np.mean(all_embeddings, axis=0)
