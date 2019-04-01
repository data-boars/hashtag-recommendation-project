import functools

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from tweet_recommendations.embeddings.word2vec import get_w2v_tweet_embedding, load_w2v_model
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text

CONFIG_KEYS = ['K', 'w2v_function', 'embedding_name', 'popularity_measure', 'popularity_to_similarity_ratio']


def embedding_similarity(x: np.ndarray, y: np.ndarray):
    """
    Computes angular similarity based on cosine similarity. 
    https://en.wikipedia.org/wiki/Cosine_similarity#Angular_distance_and_similarity
    Angular similarity is bound to [0, 1] and angular distance is a formal distance metric.
    """
    
    # clipping to avoid computational errors
    similarity = np.clip(cosine_similarity(x, y), -1, 1)
    ang_dist = np.arccos(similarity) / np.pi
    ang_sim = 1 - ang_dist
    return ang_sim


def normalise(array: np.ndarray):
    if not np.isclose(array.max(), array.min(), rtol=1e-8):
        return (array - array.min()) / (array.max() - array.min())
    return array


def recommend_for_embedding(embedding: np.ndarray, hashtags_df: pd.DataFrame, config: dict):
    assert all(key in config for key in CONFIG_KEYS if key != 'w2v_function')

    embedding_name = config['embedding_name']

    if 'override_hashtag_embeddings' in config:
        hashtag_embeddings = config['override_hashtag_embeddings'][embedding_name]
    else:
        hashtag_embeddings = np.asarray(hashtags_df[embedding_name].values.tolist())
    similarities = embedding_similarity(embedding.reshape(1, -1), hashtag_embeddings).reshape(-1)

    return recommend_with_computed_similarities(similarities, hashtags_df, config)


def recommend_with_computed_similarities(similarities: np.ndarray, hashtags_df: pd.DataFrame, config: dict):
    # w2v_function is no longer required in config, when we have precomputed embedding similarities
    assert all(key in config for key in CONFIG_KEYS if key != 'w2v_function'), 

    popularity_measure = config['popularity_measure']
    popularity_to_similarity_ratio = config['popularity_to_similarity_ratio']
    K = config['K']

    similarities = similarities.reshape(-1)
    sim_pop_mix = prepare_similarity_and_popularity_mix(similarities,
                                                        hashtags_df[popularity_measure].values,
                                                        popularity_to_similarity_ratio, config)

    top_k = np.argpartition(-sim_pop_mix, np.arange(K))[:K]
    top_k = hashtags_df['hashtag'].iloc[top_k]
    return list(top_k)


def prepare_similarity_and_popularity_mix(similarities: np.ndarray, popularities: np.ndarray,
                                          popularity_to_similarity_ratio: float, config: dict):
    similarities = normalise(similarities)
    popularities = normalise(popularities)

    if 'similarity_popularity_mix_function' in config:
        sim_pop_mix = config['similarity_popularity_mix_function'](similarities, popularities)
    else:
        sim_pop_mix = ((1 - popularity_to_similarity_ratio) * similarities
                       + (popularity_to_similarity_ratio * popularities))
    return sim_pop_mix


def recommend_with_config(text: str, hashtags_df: pd.DataFrame, config: dict):
    if 'is_input_embedding' in config and config['is_input_embedding']:
        embedding = text
    else:
        assert all(key in config for key in CONFIG_KEYS)
        w2v_function = config['w2v_function']
        lemmas = get_wcrft2_results_for_text(text)
        embedding = w2v_function(lemmas)
    return recommend_for_embedding(embedding, hashtags_df, config)


def prepare_base_config(w2v_model_path: str = '/mnt/SAMSUNG/models/embeddings/kgr10.plain.skipgram.dim100.neg10.vec'):
    conf = {'K': 10,
            'w2v_function': functools.partial(get_w2v_tweet_embedding,
                                              w2v_model=load_w2v_model(w2v_model_path)),
            'embedding_name': 'skipgram',
            'popularity_measure': 'pagerank',
            'popularity_to_similarity_ratio': 0.2}

    return conf
