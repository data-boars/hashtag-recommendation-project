from collections import namedtuple, Counter
from itertools import chain
from typing import Optional, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.neighbors.kd_tree import KDTree

from tweet_recommendations.estimator import Estimator

CentroidData = namedtuple("CentroidData", ["cluster_medoid", "popular_hashtags"])


class DBScanBasedEstimator(Estimator):
    def __init__(self,
                 verbose: bool = False):
        self._clusters: np.ndarray = None

        self.clusterizer = None
        self.neighbours = None
        self.centroids_data = {}

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
        :return: self.
        """
        minimal_hashtag_occurence = fit_params["minimal_hashtag_occurence"]

        if self.verbose:
            print(f"Data input shape: {x.shape}")

        x = self.drop_tweets_with_hashtags_that_occurred_less_than(x,
                                                                   minimal_hashtag_occurence)
        embeddings = np.array(x["embedding"].to_list())

        if self.verbose:
            print(f"Data embedding shape, after "
                  f"droping data by given criteria: {embeddings.shape}")

        self.neighbours = KDTree(embeddings, metric="euclidean")
        epsilon = np.mean(self.neighbours.query(embeddings, k=2)[0][:, 1])

        if self.verbose:
            print(f"Found epsilon: {epsilon}")

        self.clusterizer = DBSCAN(metric='manhattan', eps=epsilon, min_samples=1)
        x["cluster_label"] = self.clusterizer.fit_predict(embeddings)

        if self.verbose:
            print("Clustering finished.")

        x["hashtags"] = x["hashtags"].apply(lambda r: [elem["text"] for elem in r])

        grouped_clusters = x.groupby(["cluster_label"])

        if self.verbose:
            print("Clusters grouped. Building cluster data")

        for cluster_number, cluster in grouped_clusters:
            cluster_embeddings = np.array(cluster["embedding"].to_list())
            cluster_center = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            kd_tree = KDTree(cluster_embeddings, metric="euclidean")
            cluster_medoid_index = kd_tree.query(cluster_center, k=1)[1][0]
            cluster_medoid = cluster_embeddings[cluster_medoid_index]
            aggregated_hashtags = list(chain.from_iterable(cluster["hashtags"].values))
            sorted_most_popular_hashtags = [hashtag for hashtag, count in
                                            Counter(aggregated_hashtags).most_common()]
            self.centroids_data[cluster_number] = CentroidData(
                cluster_medoid=cluster_medoid,
                popular_hashtags=sorted_most_popular_hashtags)

        return self

    def transform(self, x: np.ndarray) -> List[List[str]]:
        """
        For a given tweet/tweets embeddings recommend hashtags.
        :param x: ndarray of tweets embeddings.
        :return: Iterable of recommended hashtags.
        """
        recommended_hashtags = []
        x = x if len(x.shape) == 2 else x.reshape(1, -1)
        tweets_number = x.shape[0]
        for i in range(tweets_number):
            min_dist = np.inf
            best_hashtags = []
            for centroid_id, centroid_data in self.centroids_data.items():
                current_distance = cosine(centroid_data.cluster_medoid, x[i, :])
                if min_dist > current_distance:
                    min_dist = current_distance
                    best_hashtags = centroid_data.popular_hashtags
            recommended_hashtags.append(best_hashtags)
        return recommended_hashtags
