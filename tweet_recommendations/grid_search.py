import os
import pandas as pd
import numpy as np
from itertools import product
import pickle
import dask.bag as db
from dask.diagnostics import ProgressBar
from contextlib import suppress

from tweet_recommendations.utils.metrics import (get_average_precision_at_k, get_rank_dcg_at_k, get_recall_at_k)
from tweet_recommendations.recommendation import recommend_with_computed_similarities, embedding_similarity
from tweet_recommendations.datasets_access import (get_hashtags_df,
                                                   add_hashtag_occurences_counts,
                                                   get_full_dataset, get_train_dataset,
                                                   get_val_dataset, get_test_dataset,
                                                   filter_hashtags_by_occurences_limit,
                                                   compute_expected_hashtags_with_respect_to_hashtags_df)


METRICS = {'recall': get_recall_at_k,
           'mAP': get_average_precision_at_k,
           'ndcg': get_rank_dcg_at_k}


EMBEDDING_NAMES = ['skipgram', 'fasttext']
POPULARITY_MEASURES = ['pagerank', 'mean_retweets']
RATIOS = np.linspace(0, 1, 11)
Ks = [5, 10, 15, 20, 30, 40, 50, 75, 100]
HASHTAG_OCCURENCE_COUNT_LIMIT = [0, 1, 5, 10, 20]


def create_datasets_pool():
    full_dataset = get_full_dataset()
    train_df, pop_train_df, unpop_train_df = get_train_dataset(full_dataset)
    val_df, pop_val_df, unpop_val_df = get_val_dataset(full_dataset)
    test_df, pop_test_df, unpop_test_df = get_test_dataset(full_dataset)

    return {'train': train_df,
            'train_popular': pop_train_df,
            'train_unpopular': unpop_train_df,
            'val': val_df,
            'val_popular': pop_val_df,
            'val_unpopular': unpop_val_df,
            'test': test_df,
            'test_popular': pop_test_df,
            'test_unpopular': unpop_test_df}


def build_hyperparameters_dataframe(dataset_names, embeddings, popularities, ratios, ks, count_limits, ):
    return pd.DataFrame(list(product(dataset_names, embeddings, popularities, ratios, ks, count_limits)),
                        columns=['dataset', 'embedding', 'popularity', 'ratio', 'K', 'occurence_limit'])


def get_metrics_for_recommendation(expected, recommended, metrics_functions, k):
    return {metric: metrics_functions[metric](expected, recommended, k)
            for metric in metrics_functions}


def experiment(tweets_df, hashtags_df, hyperparameters, metrics_to_calculate=METRICS, backup_dir=None):
    config = {'popularity_to_similarity_ratio': hyperparameters['ratio'],
              'embedding_name': hyperparameters['embedding'],
              'popularity_measure': hyperparameters['popularity'],
              'K': hyperparameters['K']}

    embedding = hyperparameters['embedding']
    hashtags = filter_hashtags_by_occurences_limit(hashtags_df, hyperparameters['occurence_limit'])
    df = compute_expected_hashtags_with_respect_to_hashtags_df(tweets_df, hashtags)

    precomputed_similarities = embedding_similarity(np.asarray(df[embedding].values.tolist()),
                                                    np.asarray(hashtags[embedding].values.tolist()))
    df['similarities'] = [precomputed_similarities[i] for i in range(len(precomputed_similarities))]

    df['recommendations'] = df['similarities'].apply(
        lambda sim: recommend_with_computed_similarities(sim, hashtags, config))

    metrics = df.apply(lambda row: get_metrics_for_recommendation(row['expected_hashtags'], row['recommendations'],
                                                                  metrics_to_calculate, config['K']),
                       axis=1)
    result = pd.DataFrame(list(metrics)).mean()

    if backup_dir is not None:
        bak = {col: result[col] for col in result.index}
        fname = '_'.join([str(x) for x in hyperparameters.values()]) + '.pkl'
        with open(backup_dir + '/' + fname, 'wb') as f:
            pickle.dump({**hyperparameters, **bak}, f)

    return result


def create_progressbar():
    return ProgressBar()


def grid_search(datasets_pool, hashtags_df, hyperparameters_df, metrics_to_calc=METRICS, backup_dir=None, progressbar=suppress()):
    if backup_dir is not None:
        os.makedirs(backup_dir, exist_ok=True)

    bag = db.from_sequence(hyperparameters_df.to_dict('records'))

    if 'count' not in hashtags_df:
        hashtags_df = add_hashtag_occurences_counts(hashtags_df, datasets_pool['train'])

    with progressbar:
        result = bag.map(lambda hyperparameters: experiment(datasets_pool[hyperparameters['dataset']],
                                                            hashtags_df,
                                                            hyperparameters,
                                                            metrics_to_calc,
                                                            backup_dir=backup_dir)).compute()
    return pd.concat([hyperparameters_df, pd.DataFrame(result)], axis=1)
