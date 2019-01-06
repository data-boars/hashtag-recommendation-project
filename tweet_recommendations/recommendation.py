from tweet_recommendations.embeddings.fast_text import get_fasttext_tweet_embedding, load_fasttext_model
from tweet_recommendations.embeddings.word2vec import get_w2v_tweet_embedding, load_w2v_model
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import functools


CONFIG_KEYS = ['K', 'w2v_function', 'embedding_name', 'popularity_measure', 'popularity_to_similarity_ratio']


def embedding_similarity(x, y):
    similarity = cosine_similarity(x, y)
    ang_dist = np.arccos(similarity) / np.pi
    ang_sim = 1 - ang_dist
    return ang_sim


def normalise(array):
    return (array - array.min()) / (array.max() - array.min())


def recommend_for_embedding(embedding, hashtags_df, config):
    assert all(key in config for key in CONFIG_KEYS if key != 'w2v_function')

    embedding_name = config['embedding_name']
    popularity_measure = config['popularity_measure']
    popularity_to_similarity_ratio = config['popularity_to_similarity_ratio']
    K = config['K']

    hashtag_embeddings = np.vstack(hashtags_df[embedding_name])
    similarities = embedding_similarity(embedding.reshape(1, -1), hashtag_embeddings).reshape(-1)

    sim_pop_mix = prepare_similarity_and_popularity_mix(similarities, hashtags_df[popularity_measure].values,
                                                        popularity_to_similarity_ratio, config)

    top_k = sim_pop_mix.argsort()[-K:][::-1]
    top_k = hashtags_df['hashtag'].iloc[top_k]
    return list(top_k)


def prepare_similarity_and_popularity_mix(similarities, popularities, popularity_to_similarity_ratio, config):
    similarities = normalise(similarities)
    popularities = normalise(popularities)

    if 'similarity_popularity_mix_function' in config:
        sim_pop_mix = config['similarity_popularity_mix_function'](similarities, popularities)
    else:
        sim_pop_mix = similarities + (popularity_to_similarity_ratio * popularities)
    return sim_pop_mix


def recommend_with_config(text, hashtags_df, config):
    if 'is_input_embedding' in config or config['is_input_embedding']:
        embedding = text
    else:
        assert all(key in config for key in CONFIG_KEYS)
        w2v_function = config['w2v_function']
        lemmas = get_wcrft2_results_for_text(text)
        embedding = w2v_function(lemmas)
    return recommend_for_embedding(embedding, hashtags_df, config)


def prepare_base_config(w2v_model_path='/mnt/SAMSUNG/models/embeddings/kgr10.plain.skipgram.dim100.neg10.vec'):
    conf = {'K': 10,
            'w2v_function': functools.partial(get_w2v_tweet_embedding,
                                              w2v_model=load_w2v_model(w2v_model_path)),
            'embedding_name': 'skipgram',
            'popularity_measure': 'pagerank',
            'popularity_to_similarity_ratio': 0.2}

    return conf
