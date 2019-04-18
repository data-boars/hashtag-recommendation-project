from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def load_w2v_model(w2v_model_path: str) -> KeyedVectors:
    return KeyedVectors.load(w2v_model_path, mmap="r")


def get_w2v_tweets_embeddings(twitter_data: List[Tuple[str, List[str], str]], w2v_model: KeyedVectors) -> pd.DataFrame:
    id_list = []
    meaned_word_vector_list = []
    for tweet_id, tweet_word_list, status in twitter_data:
        id_list.append(tweet_id)
        meaned_word_vector_list.append(
            get_w2v_tweet_embedding(tweet_word_list, w2v_model)
        )
    return pd.DataFrame(
        data={"tweet_id": id_list, "tweet_embedding": meaned_word_vector_list}
    )


def get_w2v_tweet_embedding(tweet_word_list: List[str], w2v_model: KeyedVectors) -> np.ndarray:
    all_embeddings = []
    total_words = 0
    error_words = 0
    for word in tweet_word_list:
        try:
            word_embedding = w2v_model[word.lower()]
            all_embeddings.append(word_embedding)
        except Exception:
            error_words += 1
        total_words += 1
    print("Percentage of words not found: {:.4f}".format(error_words / total_words))
    return np.mean(all_embeddings, axis=0)
