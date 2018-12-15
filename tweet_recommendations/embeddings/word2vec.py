import numpy as np
import pandas as pd
from gensim.models import KeyedVectors

from tweet_recommendations.utils.constants import WORD2VEC_MODEL_PATH


def get_tweets_embeddings(twitter_data):
    id_list = []
    meaned_word_vector_list = []
    for tweet_id, tweet_word_list, status in twitter_data:
        id_list.append(tweet_id)
        meaned_word_vector_list.append(get_tweet_embedding(tweet_word_list))
    return pd.DataFrame(
        data={"tweet_id": id_list, "tweet_embedding": meaned_word_vector_list}
    )


def get_tweet_embedding(tweet_word_list):
    model = KeyedVectors.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=False)
    meaned_word_vector = np.zeros(100)
    count_valid_embeddings = 0
    for word in tweet_word_list:
        try:
            word_embedding = model[word.lower()]
            meaned_word_vector = np.add(word_embedding, meaned_word_vector)
            count_valid_embeddings += 1
        except Exception:
            count_valid_embeddings -= 1
            print("Key '{}' not found in w2v dict".format(word))
    meaned_word_vector /= len(tweet_word_list)
    del model
    return meaned_word_vector
