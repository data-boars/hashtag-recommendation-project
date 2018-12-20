from typing import List, Tuple

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors


def get_w2v_tweets_embeddings(
    twitter_data: List[Tuple(str, List[str], str)], w2v_model_path: str
) -> pd.DataFrame:
    id_list = []
    meaned_word_vector_list = []
    for tweet_id, tweet_word_list, status in twitter_data:
        id_list.append(tweet_id)
        meaned_word_vector_list.append(
            get_w2v_tweet_embedding(tweet_word_list, w2v_model_path)
        )
    return pd.DataFrame(
        data={"tweet_id": id_list, "tweet_embedding": meaned_word_vector_list}
    )


def get_w2v_tweet_embedding(
    tweet_word_list: List[str], w2v_model_path: str
) -> np.ndarray:
    model = KeyedVectors.load_word2vec_format(w2v_model_path, binary=False)
    all_embeddings = []
    for word in tweet_word_list:
        try:
            word_embedding = model[word.lower()]
            all_embeddings.append(word_embedding)
        except Exception:
            print("Key '{}' not found in w2v dict".format(word))
    del model
    return np.mean(all_embeddings, axis=0)
