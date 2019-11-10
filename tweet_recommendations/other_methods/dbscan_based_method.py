import multiprocessing as mp
from collections import Counter
from itertools import chain
from typing import Tuple, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from scipy.spatial import cKDTree as KDTree, distance
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

from tweet_recommendations.embeddings.sif_embedding import SIFEmbedding
from tweet_recommendations.other_methods.method import Method

# "Finally, we use one as the number of minPoints and the average value of this vector as epsilon."
# according to publication, this value should be fixed to 1.
MIN_SAMPLES = 1


class DBScanBasedEstimator(Method):
    def __init__(self, path_to_keyedvectors_model: Optional[str] = None, verbose: bool = False):
        """
        Hashtag recommendation method implementation based on https://bit.ly/2V3tnWX.
        :param path_to_keyedvectors_model: Path to converted by script `convert_embedding_model_to_mmap.py` gensim
        model. It can either word2vec or fasttext, `gensim` handles both.
        :param verbose: Whether method should be verbose
        """
        self._clusters: np.ndarray = None

        self.clusterizer = None
        self.sif_embedding = SIFEmbedding(path_to_keyedvectors_model, verbose=verbose)
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
        neighbours_count = fit_params.get("neighbours_count", 16)

        if self.verbose:
            print(f"Data input shape: {x.shape}")

        x = self.drop_tweets_with_hashtags_that_occurred_less_than(x, minimal_hashtag_occurence)

        if self.verbose:
            print("Setup tweet embedding method")
            print("Tweet content lemmas embedding started")

        embeddings = self.sif_embedding.fit_transform(x, None)
        x["embedding"] = pd.Series(embeddings.tolist())

        if self.verbose:
            print(f"Data embedding shape, after droping data by given criteria: {embeddings.shape}")

        self.neighbours = KDTree(embeddings)

        epsilon = np.mean(self.neighbours.query(embeddings, k=neighbours_count, n_jobs=mp.cpu_count(), p=1)[0])

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

            # matrix of shape (1 x N) is returned
            distances = distance.cdist(cluster_center, cluster_embeddings, metric="cityblock")[0]

            cluster_centroid_index = np.argmin(
                distances)  # we are sure that it will not point to itself because mean center should be not existent

            cluster_centroid = cluster_embeddings[cluster_centroid_index]
            aggregated_hashtags = list(chain.from_iterable(cluster["hashtags"].to_numpy()))
            sorted_most_popular_hashtags = [hashtag for hashtag, count in Counter(aggregated_hashtags).most_common()]

            self._centroids_data.append(cluster_centroid)
            self._corresponding_to_centroids_data_hashtags.append(sorted_most_popular_hashtags)

        self._centroids_data = np.vstack(self._centroids_data)
        self._corresponding_to_centroids_data_hashtags = np.asarray(self._corresponding_to_centroids_data_hashtags)

        return self

    def transform(self, x: Union[Tuple[Tuple[str, ...], ...], Tuple[str, ...]], **kwargs) -> np.ndarray:
        """
        For a given tweet/tweets embeddings recommend hashtags.
        :param x: tuple of tuple of str or tuple of str. If first argument of x is a tuple is str, it is assumed that
            tuple contains already lemmatized text. If single str is present as first element, it is assumed
            that lemmatization has to be performed.
        :return: Iterable of recommended hashtags.
        """
        embeddings = self.sif_embedding.transform(x)
        embeddings = embeddings if len(embeddings.shape) == 2 else embeddings.reshape(1, -1)
        distances = cosine_distances(embeddings, self._centroids_data)
        sorted_distances_indices = np.argsort(distances, axis=1)
        recommended_hashtags = self._corresponding_to_centroids_data_hashtags[sorted_distances_indices]
        result = self.post_process_result(recommended_hashtags)

        return result

    @classmethod
    def post_process_result(cls, recommended_hashtags: np.ndarray) -> np.ndarray:
        """
        Flattens lists of recommended hashtags for each tweet. Two level list is leftover from the `fit` because tags
        inside a single cluster are sorted according to their popularity.
        :param recommended_hashtags: Two level list of recommended hashtags for each tweet and each tweet has
            clusters of tags.
        :return: np.ndarray of flattened list of recommended tags.
        """
        result = []
        for tweet_tags in recommended_hashtags:
            result.append([tag for centroid_tags in tweet_tags for tag in centroid_tags])
        return np.asarray(result)
