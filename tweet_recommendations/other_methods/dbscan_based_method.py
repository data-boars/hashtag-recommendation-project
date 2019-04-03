from typing import *

import numpy as np
import pandas as pd
import tqdm
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import paired_manhattan_distances

from tweet_recommendations.utils.clients import get_wcrft2_results_for_text
from ..estimator import Estimator


class DBScanBasedEstimator(Estimator):
    def __init__(self,
                 vectorizing_model: Estimator,
                 verbose: bool = False):
        self._hashtag_occurrences: np.ndarray = None
        self._hashtag_representations: np.ndarray = None
        self._clusters: np.ndarray = None

        self.vectorizing_model = vectorizing_model
        self.clusterizer = DBSCAN(metric=paired_manhattan_distances)

        self.verbose = verbose

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params):
        """
        Builds hashtags representations using w2v method and clusters them using DBScan.
        :param x: pd.DataFrame with tweet content, user id, and separate hashtags. It is "original_tweets.p" in our
            case.
        :param y: None, needed for compatibility.
        :param fit_params:
            minimal_hashtag_occurrence: int. If hashtag occurred less than this number then it's not considered during
                prediction (simply removed). To include all hashtags put number <= 0.
        :return: self.
        """
        minimal_hashtag_occurence = fit_params["minimal_hashtag_occurence"]

        x = self.drop_hashtag_that_occurred_less_than(x, minimal_hashtag_occurence)

        if self.verbose:
            print("Creating embeddings ...")
            tqdm.tqdm.pandas()
            x["embedding"] = x.progress_apply(lambda r: self.vectorizing_model.transform(r["lemmas"]))
        else:
            x["embedding"] = x.apply(lambda r: self.vectorizing_model.transform(r["lemmas"]))

        x["hashtags"] = x["hashtags"].apply(lambda r: [elem["text"] for elem in r["hashtags"]])
        x["embedding_labels"] = self.clusterizer.fit_predict(x["embedding"])

        tags_labels_embeddings = x[["hashtags", "embedding_label", "embedding"]]
        tags_labels_embeddings["hashtags"].apply(pd.Series) \
            .merge(tags_labels_embeddings, right_index=True, left_index=True) \
            .drop(["hashtags"], axis=1) \
            .melt(id_vars=["id_str", "hashtags"], value_name="hashtag") \
            .drop("variable", axis=1) \
            .dropna()

        return self

    def transform(self, x: List[str]):
        """
        For a given tweet represented as a list of lemmas recommends hashtags.
        :param x: list of str or str. If list is str, strs are lemmas of the tweet. If single str, it is assumed that
            lemmatization has to be performed.
        :return: Iterable of recommended hashtags.
        """
        if isinstance(x, str):
            x = get_wcrft2_results_for_text(x)
        if isinstance(x, list):
            x = ' '.join(x)
        pass
