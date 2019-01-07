from typing import List, Tuple

import numpy as np
import pandas as pd

import fastText


def load_fasttext_model(model_path: str):
    return fastText.load_model(model_path)


def get_fasttext_tweets_embeddings(
    twitter_data: List[Tuple[str, List[str], str]], fasttext_model: fastText.FastText
) -> pd.DataFrame:
    id_list = []
    meaned_word_vector_list = []
    for tweet_id, tweet_word_list, status in twitter_data:
        id_list.append(tweet_id)
        meaned_word_vector_list.append(
            get_fasttext_tweet_embedding(tweet_word_list, fasttext_model)
        )
    return pd.DataFrame(
        data={"tweet_id": id_list, "tweet_embedding": meaned_word_vector_list}
    )


def get_fasttext_tweet_embedding(
    tweet_word_list: List[str], fasttext_model: fastText.FastText
) -> np.ndarray:
    all_embeddings = []
    for word in tweet_word_list:
        all_embeddings.append(fasttext_model.get_word_vector(word))
    return np.mean(all_embeddings, axis=0)
