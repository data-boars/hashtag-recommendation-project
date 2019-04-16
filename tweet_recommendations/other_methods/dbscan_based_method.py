from collections import namedtuple, Counter
from itertools import chain
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.neighbors.kd_tree import KDTree

from tweet_recommendations.other_methods.method import Method

CentroidData = namedtuple("CentroidData", ["cluster_medoid", "popular_hashtags"])
# "Finally, we use one as the number of minPoints and the average value of this vector as epsilon."
# according to publication, this value should be fixed to 1.
MIN_SAMPLES = 1


class DBScanBasedEstimator(Method):
    def __init__(self, embedding_method: Method, verbose: bool = False):
        self._clusters: np.ndarray = None

        self.clusterizer = None
        self.embedding_method = embedding_method
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
            neighbours_count: int. Parameter needed to calculate epsilon param for DBSCAN method.
            Min param value 2, max param value embedding data count.
        :return: self.
        """
        minimal_hashtag_occurence = fit_params.get("minimal_hashtag_occurence", 10)
        neighbours_count = fit_params.get("neighbours_count", 2)

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

        self.neighbours = KDTree(embeddings, metric="euclidean")
        nearest_neighbors_distances = \
            self.neighbours.query(embeddings, k=neighbours_count)[0]
        epsilon = np.mean(nearest_neighbors_distances[:, 1:])

        if self.verbose:
            print(f"Found epsilon: {epsilon}")

        self.clusterizer = DBSCAN(metric='manhattan', eps=epsilon,
                                  min_samples=MIN_SAMPLES)
        x["cluster_label"] = self.clusterizer.fit_predict(embeddings)

        if self.verbose:
            print("Clustering finished.")

        x["hashtags"] = x["hashtags"].apply(lambda r: [elem["text"] for elem in r])

        grouped_clusters = x.groupby(["cluster_label"])

        if self.verbose:
            print("Clusters grouped. Building cluster data")

        for cluster_number, cluster in grouped_clusters:
            cluster_embeddings = np.vstack(cluster['embedding'].to_numpy())
            cluster_center = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
            kd_tree = KDTree(cluster_embeddings, metric="euclidean")
            cluster_medoid_index = kd_tree.query(cluster_center, k=1)[1][0]
            cluster_medoid = cluster_embeddings[cluster_medoid_index]
            aggregated_hashtags = list(
                chain.from_iterable(cluster["hashtags"].to_numpy()))
            sorted_most_popular_hashtags = [hashtag for hashtag, count in
                                            Counter(aggregated_hashtags).most_common()]
            self.centroids_data[cluster_number] = CentroidData(
                cluster_medoid=cluster_medoid,
                popular_hashtags=sorted_most_popular_hashtags)

        return self

    def transform(self, x: Union[List[str], str]) -> List[List[str]]:
        """
        For a given tweet/tweets embeddings recommend hashtags.
        :param x: ndarray of tweets embeddings.
        :return: Iterable of recommended hashtags.
        """
        recommended_hashtags = []
        embeddings = self.embedding_method.transform(x)
        embeddings = embeddings if len(embeddings.shape) == 2 else embeddings.reshape(1,
                                                                                      -1)
        tweets_number = embeddings.shape[0]
        for i in range(tweets_number):
            min_dist = np.inf
            best_hashtags = []
            for centroid_id, centroid_data in self.centroids_data.items():
                current_distance = cosine(centroid_data.cluster_medoid,
                                          embeddings[i, :])
                if min_dist > current_distance:
                    min_dist = current_distance
                    best_hashtags = centroid_data.popular_hashtags
            recommended_hashtags.append(best_hashtags)
        return recommended_hashtags


def drop_tweets_with_hashtags_that_occurred_less_than(data: pd.DataFrame,
                                                      minimal_hashtag_occurrence: int) -> \
        List[str]:
    hashtags = data["hashtags"].tolist()
    hashtags = [h['text'] for a_list in hashtags for h in a_list]
    counts = Counter(hashtags)
    filtered_tags = [t for t, count in counts.items() if
                     count >= minimal_hashtag_occurrence]
    return filtered_tags


def drop_tweets_which_not_contain_given_hashtags(data: pd.DataFrame,
                                                 filtered_tags: List[
                                                     str]) -> pd.DataFrame:
    data["hashtags"] = data["hashtags"].apply(
        lambda x: [elem for elem in x if elem["text"] in filtered_tags])
    data = data.drop(data[data["hashtags"].str.len() == 0].index)
    return data
