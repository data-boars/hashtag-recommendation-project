import pickle
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
import scipy.spatial

from tweet_recommendations.data_processing.data_loader import convert_hashtags_dicts_to_list


def _dummy_progressbar(iterable: Iterable, **kwargs):
    return iterable


def build_base_tweets_graph(tweets_df: pd.DataFrame, progress_bar=_dummy_progressbar):

    assert "id" in tweets_df, "Column 'id' not present in tweets_df"
    assert "hashtags" in tweets_df, "Column 'hashtags' not present in tweets_df"
    assert "retweet_count" in tweets_df, "Column 'retweet_count' not present in tweets_df"
    assert tweets_df["hashtags"].apply(lambda x: isinstance(x, list)).all(), "Column 'hashtags' doesn't contain lists"
    assert (tweets_df["hashtags"]
            .apply(lambda x: all(isinstance(elem, str) for elem in x))
            .all()
           ), "Not all elements of 'hashtags' lists are strings"

    G = nx.Graph()
    for row in progress_bar(tweets_df.itertuples(), total=len(tweets_df)):
        if row.hashtags:
            G.add_node(row.id)
            G.node[row.id]["node_type"] = "tweet"
            G.node[row.id]["retweets"] = row.retweet_count
            for hashtag in row.hashtags:
                G.add_node(hashtag)
                G.node[hashtag]["node_type"] = "hashtag"
                G.add_edge(row.id, hashtag)

    return G


def add_tweet_embeddings_to_graph(G: nx.Graph, embeddings_df: pd.DataFrame,
                                  embedding_name: str = 'embedding'):
    assert 'id' in embeddings_df, "Column 'id' not present in embeddings_df"
    assert 'embedding' in embeddings_df, "Column 'embedding' not present in embeddings_df"

    embeddings_dict = embeddings_df.set_index('id').to_dict()['embedding']
    nx.set_node_attributes(G, embeddings_dict, embedding_name)
    return G


def calculate_hashtag_embeddings(G: nx.Graph, embedding_name: str = 'embedding', progress_bar=_dummy_progressbar):
    for node in progress_bar(G.nodes, total=len(G.nodes)):
        if G.nodes[node]["node_type"] == "hashtag":
            tweets = G.neighbors(node)
            embeddings = np.asarray([G.node[tweet][embedding_name] for tweet in tweets])
            G.node[node][embedding_name] = embeddings.mean(axis=0)

    for node in G.nodes:
        assert embedding_name in G.node[node], f"Node '{node}' doesn't have an attribute: 'embedding name'"

    return G


def calculate_edge_weights(G: nx.Graph,
                           embedding_name: str = 'embedding',
                           distance_name: str = 'distance',
                           similarity_name: str = 'similarity',
                           progress_bar=_dummy_progressbar):
    for node_from, node_to, edge_features in progress_bar(G.edges(data=True), total=len(G.edges)):
        emb_from = G.node[node_from][embedding_name]
        emb_to = G.node[node_to][embedding_name]

        distance = scipy.spatial.distance.cosine(emb_from, emb_to)
        similarity = 1 - distance
        ang_dist = np.arccos(similarity) / np.pi
        ang_sim = 1 - ang_dist

        edge_features[distance_name] = ang_dist
        edge_features[similarity_name] = ang_sim

    return G


def calculate_pagerank(G: nx.Graph):
    graph_pagerank = nx.pagerank(G)
    nx.set_node_attributes(G, graph_pagerank, "pagerank")

    return G


def calculate_hashtag_popularity_mean_retweets_heuristic(G: nx.Graph, progress_bar=_dummy_progressbar):
    for node in progress_bar(G.nodes, total=len(G.nodes)):
        if G.nodes[node]["node_type"] == "hashtag":
            tweets = G.neighbors(node)
            retweets_counts = np.asarray([G.nodes[tweet]['retweets'] for tweet in tweets])
            G.nodes[node]['mean_retweets'] = retweets_counts.mean(axis=0)

    for node in G.nodes:
        assert 'mean_retweets' in G.nodes[node] if G.nodes[node]['node_type'] == 'hashtag' else True,\
        f"Node '{node}' doesn't have an attribute: 'mean_retweets'"

    return G


def build_graph_pipeline(tweets_df, embeddings_df, progress_bar=None):

    assert "embedding" in embeddings_df, "Column 'embedding' not present in embeddings_df"
    assert "id" in embeddings_df, "Column 'id' not present in embeddings_df"
    assert "hashtags" in tweets_df, "Column 'hashtags' not present in tweets_df"
    assert "id" in tweets_df, "Column 'id' not present in tweets_df"

    tweets_with_tags = tweets_df["hashtags"][tweets_df["hashtags"].str.len() > 0]
    if tweets_with_tags.apply(lambda tags: all(isinstance(x, dict) for x in tags)).all():
        tweets_df = convert_hashtags_dicts_to_list(tweets_df)

    df = tweets_df.merge(embeddings_df, on="id")

    G = build_base_tweets_graph(df, progress_bar)
    G = calculate_hashtag_embeddings(G, progress_bar)
    G = calculate_edge_weights(G, progress_bar)
    G = calculate_pagerank(G)
    return G


if __name__ == "__main__":
    tweets_df = pd.read_pickle("../../data/source_data/original_tweets.p")
    tweets_df["hashtags"] = tweets_df["hashtags"].apply(lambda x: [item["text"] for item in x])
    embeddings_df = pd.read_pickle("../../data/embeddings/embeddings.pkl")
    embeddings_df["tweet_id"] = embeddings_df["tweet_id"].astype(np.int64)
    embeddings_df = embeddings_df.rename({"embeddings": "embedding"}, axis="columns")

    df = tweets_df.merge(embeddings_df, left_on="id", right_on="tweet_id")

    G = build_base_tweets_graph(df, "text")
    G = calculate_hashtag_embeddings(G, "text")
    G = calculate_edge_weights(G, "text")
    G = calculate_pagerank(G)

    with open("./data_processing/graphs/graph.p", "wb") as f:
        pickle.dump(G, f)
