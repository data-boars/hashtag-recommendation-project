import multiprocessing as mp
from collections import Counter
from itertools import chain
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import cKDTree as KDTree, distance
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from tweet_recommendations.other_methods.method import Method

# "Finally, we use one as the number of minPoints and the average value of this vector as epsilon."
# according to publication, this value should be fixed to 1.
MIN_SAMPLES = 1


class DBScanBasedEstimator(Method):
    def __init__(self, embedding_method: Method, verbose: bool = False):
        self._clusters: np.ndarray = None

        self.clusterizer = None
        self.embedding_method = embedding_method
        self.neighbours = None
        self._centroids_data = []
        self._corresponding_to_centroids_data_hashtags = []

        self.verbose = verbose

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params):
        """
        Builds hashtags representations using w2v method and clusters them using DBScan.
        :param x: pd.DataFrame with tweet content, user id, and separate hashtags.
                It is "original_tweets.p" in our case.
        :param y: None, needed for compatibility.
        :param fit_params:
            minimal_hashtag_occurrence: int. If hashtag occurred less than this number
            then it's not considered during prediction (simply removed).
            To include all hashtags put number <= 0.
            neighbours_count: int. Parameter needed to calculate epsilon param for DBSCAN method.
            Min param value 2, max param value embedding data count.
        :return: self.
        """
        minimal_hashtag_occurence = fit_params.get("minimal_hashtag_occurence", 10)

        # the value below is so high because otherwise the `epsilon` basing on this would be so small that there would
        # as many clusters as there are samples in dataset
        neighbours_count = fit_params.get("neighbours_count", 256)

        if self.verbose:
            print(f"Data input shape: {x.shape}")

        valid_hashtags = drop_tweets_with_hashtags_that_occurred_less_than(x,
                                                                           minimal_hashtag_occurence)
        x = drop_tweets_which_not_contain_given_hashtags(x, valid_hashtags)

        if self.verbose:
            print("Setup tweet embedding method")

        self.embedding_method.fit(x)

        if self.verbose:
            print("Tweet content lemmas embedding started")

        embeddings = self.embedding_method.transform(x["lemmas"])

        if self.verbose:
            print(f"Data embedding shape, after "
                  f"droping data by given criteria: {embeddings.shape}")

        self.neighbours = KDTree(embeddings)

        epsilon = np.mean(self.neighbours.query(embeddings, k=neighbours_count, n_jobs=mp.cpu_count())[0])

        if self.verbose:
            print(f"Found epsilon: {epsilon}")
            print("Fitting DBSCAN ... ")

        self.clusterizer = DBSCAN(metric='manhattan', eps=epsilon, min_samples=MIN_SAMPLES, n_jobs=mp.cpu_count())
        x["cluster_label"] = self.clusterizer.fit_predict(embeddings)

        if self.verbose:
            print("Clustering finished.")

        x["hashtags"] = x["hashtags"].apply(lambda r: [elem["text"] for elem in r])

        grouped_clusters = x.groupby(["cluster_label"])

        if self.verbose:
            print(f"Samples grouped into {len(np.unique(self.clusterizer.labels_))} clusters")
            print("Building clusters' data")

        self._centroids_data = []
        self._corresponding_to_centroids_data_hashtags = []

        for cluster_number, cluster in tqdm.tqdm(grouped_clusters, total=len(np.unique(x["cluster_label"])),
                                                 disable=not self.verbose):
            cluster_embeddings = np.vstack(cluster["embedding"].to_numpy())
            cluster_center = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            distances = distance.cdist(cluster_center, cluster_embeddings)[0]  # matrix of shape (1 x N) is returned

            cluster_medoid_index = np.argmin(
                distances)  # we are sure that it will not point to itself because mean center should be not existent

            cluster_medoid = cluster_embeddings[cluster_medoid_index]
            aggregated_hashtags = list(
                chain.from_iterable(cluster["hashtags"].to_numpy()))
            sorted_most_popular_hashtags = [hashtag for hashtag, count in
                                            Counter(aggregated_hashtags).most_common()]

            self._centroids_data.append(cluster_medoid)
            self._corresponding_to_centroids_data_hashtags.append(sorted_most_popular_hashtags)

        self._centroids_data = np.vstack(self._centroids_data)
        self._corresponding_to_centroids_data_hashtags = np.asarray(self._corresponding_to_centroids_data_hashtags)

        return self

    def transform(self, x: Union[List[List[str]], List[str]]) -> np.ndarray:
        """
        For a given tweet/tweets embeddings recommend hashtags.
        :param x: list of list of str or list of str. If first argument of x is a list is str, it is assumed that list
            contains already lemmatized text. If single str is present as first element, it is assumed
            that lemmatization has to be performed.
        :return: Iterable of recommended hashtags.
        """
        embeddings = self.embedding_method.transform(x)
        embeddings = embeddings if len(embeddings.shape) == 2 else embeddings.reshape(1, -1)
        distances = cosine_distances(embeddings, self._centroids_data)
        sorted_distances_indices = np.argsort(distances, axis=1)
        recommended_hashtags = self._corresponding_to_centroids_data_hashtags[sorted_distances_indices]
        result = self.post_process_result(recommended_hashtags)

        return result

    @classmethod
    def post_process_result(cls, recommended_hashtags: np.ndarray) -> np.ndarray:
        result = []
        for tweet_tags in recommended_hashtags:
            result.append([tag for centroid_tags in tweet_tags for tag in centroid_tags])
        return np.asarray(result)


def drop_tweets_with_hashtags_that_occurred_less_than(data: pd.DataFrame, minimal_hashtag_occurrence: int) -> List[str]:
    hashtags = data["hashtags"].tolist()
    hashtags = [h['text'] for a_list in hashtags for h in a_list]
    counts = Counter(hashtags)
    filtered_tags = [t for t, count in counts.items() if
                     count >= minimal_hashtag_occurrence]
    return filtered_tags


def drop_tweets_which_not_contain_given_hashtags(data: pd.DataFrame, filtered_tags: List[str]) -> pd.DataFrame:
    data["hashtags"] = data["hashtags"].apply(
        lambda x: [elem for elem in x if elem["text"] in filtered_tags])
    data = data.drop(data[data["hashtags"].str.len() == 0].index)
    return data
