from tweet_recommendations.embeddings.fast_text import get_fasttext_tweet_embedding
from tweet_recommendations.embeddings.word2vec import get_w2v_tweet_embedding
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


use_w2v_embedding = True


def get_hashtag_rank_for_given_tweet_text(tweet: str):
    tagged_tweet_content = get_wcrft2_results_for_text(tweet)

    embeddings = (
        get_w2v_tweet_embedding(tagged_tweet_content)
        if use_w2v_embedding
        else get_fasttext_tweet_embedding(tagged_tweet_content)
    )

    hashtag_rank = []
    # hashtag_rank = perform_some_graph_rank_magic(embeddings)

    return hashtag_rank


def get_n_most_similar_hashtags_with_embedding(embedding: np.ndarray,
                                               hashtags_df: pd.DataFrame,
                                               n: int = 1,
                                               embedding_name: str = 'embedding'):

    assert embedding_name in hashtags_df
    assert 'hashtag' in hashtags_df
    hashtag_embeddings = np.vstack(hashtags_df[embedding_name].values)
    similarities = cosine_similarity(embedding.reshape(1,-1), hashtag_embeddings)
    neg_similarities = -similarities
    most_similar = np.argsort(neg_similarities).reshape(-1)[:n]
    return list(hashtags_df['hashtag'].iloc[most_similar])
