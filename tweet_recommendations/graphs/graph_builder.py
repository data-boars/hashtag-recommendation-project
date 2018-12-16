import pandas as pd
import pickle
import networkx as nx
import numpy as np
import scipy.spatial
import tqdm
from typing import *

from tweet_recommendations.data_processing.data_loader import convert_hashtags_dicts_to_list

def _create_progress_bar(what_type: Optional[str] = None):
    if what_type == 'notebook':
        return tqdm.tqdm_notebook
    elif what_type == 'text':
        return tqdm.tqdm
    else:
        return lambda iterable, total: iterable


def build_base_tweets_graph(tweets_df: pd.DataFrame, progress_bar=None):

    assert 'embedding' in tweets_df
    assert 'id' in tweets_df
    assert 'hashtags' in tweets_df
    assert tweets_df['hashtags'].apply(lambda x: isinstance(x, list)).all()
    assert tweets_df['hashtags'].apply(lambda x: all(isinstance(elem, str) for elem in x)).all()

    G = nx.Graph()
    progress_bar = _create_progress_bar(progress_bar)
    for row in progress_bar(tweets_df.itertuples(), total=len(tweets_df)):
        if row.hashtags:
            G.add_node(row.id)
            G.node[row.id]['embedding'] = row.embedding
            G.node[row.id]['node_type'] = 'tweet'
            for hashtag in row.hashtags:
                G.add_node(hashtag)
                G.node[hashtag]['node_type'] = 'hashtag'
                G.add_edge(row.id, hashtag)

    return G


def calculate_hashtag_embeddings(G: nx.Graph, progress_bar=None):
    progress_bar = _create_progress_bar(progress_bar)
    for node in progress_bar(G.nodes, total=len(G.nodes)):
        if G.nodes[node]['node_type'] == 'hashtag':
            tweets = G.neighbors(node)
            embeddings = np.asarray([G.node[tweet]['embedding'] for tweet in tweets])
            G.node[node]['embedding'] = embeddings.mean(axis=0)

    for node in G.nodes:
        assert 'embedding' in G.node[node]

    return G


def calculate_edge_weights(G: nx.Graph, progress_bar=None):

    progress_bar = _create_progress_bar(progress_bar)

    for node_from, node_to, edge_features in progress_bar(G.edges(data=True), total=len(G.edges)):
        emb_from = G.node[node_from]['embedding']
        emb_to = G.node[node_to]['embedding']

        distance = scipy.spatial.distance.cosine(emb_from, emb_to)
        similarity = 1 - distance
        ang_dist = np.arccos(similarity) / np.pi
        ang_sim = 1 - ang_dist

        edge_features['distance'] = ang_dist
        edge_features['similarity'] = ang_sim

        return G


def build_graph_pipeline(tweets_df, embeddings_df, progress_bar=None):

    assert 'embedding' in embeddings_df
    assert 'id' in embeddings_df
    assert 'hashtags' in tweets_df
    assert 'id' in tweets_df

    if isinstance(tweets_df['hashtags'][tweets_df['hashtags'].str.len()>0].iloc[0][0], dict):
        tweets_df = convert_hashtags_dicts_to_list(tweets_df)

    df = tweets_df.merge(embeddings_df, on='id')

    G = build_base_tweets_graph(df, progress_bar)
    G = calculate_hashtag_embeddings(G, progress_bar)
    G = calculate_edge_weights(G, progress_bar)
    return G


if __name__ == '__main__':
    tweets_df = pd.read_pickle('./data_processing/source_data/original_tweets.p')
    tweets_df['hashtags'] = tweets_df['hashtags'].apply(lambda x: [item['text'] for item in x])
    embeddings_df = pd.read_pickle('./data_processing/embeddings/embeddings.pkl')
    embeddings_df['tweet_id'] = embeddings_df['tweet_id'].astype(np.int64)
    embeddings_df = embeddings_df.rename({'embeddings': 'embedding'}, axis='columns')

    df = tweets_df.merge(embeddings_df, left_on='id', right_on='tweet_id')

    G = build_base_tweets_graph(df, 'text')
    G = calculate_hashtag_embeddings(G, 'text')
    G = calculate_edge_weights(G, 'text')

    with open('./data_processing/graphs/graph.p', 'wb') as f:
        pickle.dump(G, f)
