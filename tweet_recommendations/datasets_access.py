import os
import pandas as pd
import pickle
from collections import Counter


PATH = os.path.dirname(os.path.abspath(__file__)) + '/../'


def get_full_dataset():
    all_tweets = pd.read_pickle(PATH + '/data/source_data/original_tweets.p')
    all_tweets['hashtags'] = all_tweets['hashtags'].apply(lambda x: [h['text'] for h in x])
    all_tweets['expected_hashtags'] = all_tweets['hashtags']

    skipgrams = pd.read_pickle(PATH + '/data/embeddings/skipgram_embeddings.pkl')
    skipgrams['id'] = skipgrams['tweet_id'].astype(int)
    skipgrams = skipgrams.drop(columns=['tweet_id']).rename(columns={'tweet_embedding': 'skipgram'})
    fasttexts = pd.read_pickle(PATH + '/data/embeddings/fasttext_embeddings.pkl')
    fasttexts['id'] = fasttexts['tweet_id'].astype(int)
    fasttexts = fasttexts.drop(columns=['tweet_id']).rename(columns={'embeddings': 'fasttext'})

    all_tweets = all_tweets[['id', 'text', 'retweet_count', 'hashtags', 'expected_hashtags']]
    all_tweets = all_tweets.merge(skipgrams, on='id')
    all_tweets = all_tweets.merge(fasttexts, on='id')

    return all_tweets


def prepare_df(all_tweets, graph):
    tweet_ids = [node_id for node_id in graph.nodes 
                 if graph.nodes[node_id]['node_type'] == 'tweet']
    nodes = pd.DataFrame({'id': tweet_ids})
    return all_tweets.merge(nodes, on='id')


def get_specific_dataset(all_tweets, graph_path):
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)
        
    df = prepare_df(all_tweets, graph)
    popular_df = df[df['retweet_count'] > 0]
    unpopular_df = df[df['retweet_count'] == 0]
    return df, popular_df, unpopular_df


def get_train_dataset(all_tweets):
    return get_specific_dataset(all_tweets, PATH + '/data/graphs/train_graph.p')


def get_val_dataset(all_tweets):
    return get_specific_dataset(all_tweets, PATH + '/data/graphs/val_graph.p')


def get_test_dataset(all_tweets):
    return get_specific_dataset(all_tweets, PATH + '/data/graphs/test_graph.p')


def get_datasets():
    full_dataset = get_full_dataset()
    train_df, train_pop, train_unpop = get_train_dataset(full_dataset)
    val_df, val_pop, val_unpop = get_val_dataset(full_dataset)
    test_df, test_pop, test_unpop = get_test_dataset(full_dataset)
    return train_df, val_df, test_df, test_pop, test_unpop


def get_hashtags_df(graph_path='/data/graphs/train_graph.p'):
    with open(PATH + graph_path, 'rb') as f:
        G = pickle.load(f)

    hashtags = [{'hashtag': node, **G.nodes[node]}
                for node in G.nodes
                if G.nodes[node]['node_type'] == 'hashtag']
    hashtags = pd.DataFrame(hashtags)
    return hashtags


def add_hashtag_occurences_counts(hashtags_df, reference_tweets_df):
    all_hashtags = []
    _ = reference_tweets_df.hashtags.apply(lambda h: [all_hashtags.append(x) for x in set(h)])
    cnts = Counter(all_hashtags)
    cnts = pd.DataFrame(list(cnts.items()), columns=['hashtag', 'count'])

    return hashtags_df.merge(cnts, on='hashtag')


def filter_hashtags_by_occurences_limit(hashtags_df, limit):
    return hashtags_df[hashtags_df['count'] > limit]


def compute_expected_hashtags_with_respect_to_hashtags_df(tweets_df, hashtags_df):
    available_hashtags = set(hashtags_df['hashtag'])
    df = tweets_df.copy()
    df['expected_hashtags'] = df['hashtags'].apply(lambda htags: [tag for tag in htags if tag in available_hashtags])
    return df[df['expected_hashtags'].str.len() > 0]
