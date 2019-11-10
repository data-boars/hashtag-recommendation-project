import warnings
from collections import defaultdict
from operator import itemgetter
from typing import Dict, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from tweet_recommendations.method import Method
from tweet_recommendations.other_methods.utils import ModifiedOneHotEncoder
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text


class GraphSummarizationMethod(Method):
    def __init__(
        self,
        minimal_random_walk_change_difference_value: float,
        damping_factor: float,
        max_iterations: int,
        minimal_hashtag_occurrence: int = 10,
        verbose: bool = False,
    ):
        """
        Creates estimator for predicting hashtag based on graph construction
        from https://sci-hub.tw/https://ieeexplore.ieee.org/document/7300890

        :param minimal_random_walk_change_difference_value: float. Small value.
            It is used for stop criterion in random walk. If difference between
            random walk values from previous step and random walk values from
            current step is less than that value, then algorithm is stopped.
        :param damping_factor: float. 0 <= damping_factor <= 1. Probability
            between random walk and restart.
        :param max_iterations: int. max_iterations > 0. Maximal value of
            iterations to perform for random walk.
        :param minimal_hashtag_occurrence: int. If hashtag occurred less than
            this number then it's not considered during prediction (simply
            removed). To include all hashtags put number <= 0.
        :param verbose: bool. Whether method should be talkative.
        """
        self.graph: Optional[nx.Graph] = None

        self._hashtags_tf_idf_vectorizer: Optional[TfidfVectorizer] = None
        self._hashtags_tf_idf_representation: Optional[np.ndarray] = None

        self._hashtag_labels: Optional[Union[set, np.ndarray]] = None
        self._users_labels: Optional[Union[set, np.ndarray]] = None
        self._tweet_labels: Optional[Union[set, np.ndarray]] = None

        self._transition_matrix: Optional[np.ndarray] = None
        self._hashtag_encoder: ModifiedOneHotEncoder = ModifiedOneHotEncoder()

        self.minimal_random_walk_change_difference_value = (
            minimal_random_walk_change_difference_value
        )
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.minimal_hashtag_occurrence = minimal_hashtag_occurrence

    def _transform_single_row(self, hashtag_agg: Dict, row: pd.Series):
        """
        Transforms single row of pandas `original_tweets_with_lemmas.p` to
        graph. Suffixes in node names are needed due to intersection between
        hashtags and user names.

        :param hashtag_agg: Agreggation for hashtag tf-idf calculation. Each
            hashtag is represented by list of words which it occurs in.
        :param row: Single row of aforementioned dataframe.
        :return: None.
        """
        user_name = row["username"] + "_user"
        tweet_id = str(row["id"]) + "_tweet"
        tags = row["hashtags"]

        self._users_labels.add(user_name)
        self._tweet_labels.add(tweet_id)

        if not self.graph.has_node(user_name):
            self.graph.add_node(user_name, type="username")

        if not self.graph.has_node(tweet_id):
            self.graph.add_node(tweet_id, type="tweet_id")

        for hashtag_index in tags:
            tag = hashtag_index["text"] + "_tag"
            hashtag_agg[tag] += row["lemmas"]

            if not self.graph.has_node(tag):
                self.graph.add_node(tag, type="hashtag")

            if not self.graph.has_edge(tag, user_name):
                self.graph.add_edge(tag, user_name)

            if not self.graph.has_edge(tag, tweet_id):
                self.graph.add_edge(tag, tweet_id)

            self._hashtag_labels.add(tag)

    def _refine_matrix_with_additional_connections(self):
        """
        Adds edges between hashtag nodes if they share the same user.
        :return: None.
        """
        new_graph = self.graph.copy()
        for node in tqdm.tqdm(self.graph.nodes(), disable=not self.verbose):
            if self.graph.node[node]["type"] == "hashtag":
                for neighbour in self.graph.neighbors(node):
                    if self.graph.node[neighbour]["type"] == "username":
                        for other_node in self.graph.neighbors(neighbour):
                            if (
                                self.graph.node[other_node]["type"]
                                == "hashtag"
                                and not self.graph.has_edge(node, other_node)
                                and not node == other_node
                            ):
                                new_graph.add_edge(node, other_node)
        self.graph = new_graph

    def _get_binary_incidence_matrix(self):
        # needed to preserve ordering
        all_labels = np.concatenate(
            (self._hashtag_labels, self._tweet_labels, self._users_labels)
        )
        incidence_matrix = nx.adjacency_matrix(self.graph, nodelist=all_labels)

        return incidence_matrix

    def fit(self, x: pd.DataFrame, y=None, **fit_params) -> "Method":
        """
        Builds tri partite graph of Users - Hashtags - Tweets. Hashtags are
        connected if has the same user.

        :param x: pd.DataFrame with tweet content, user id, and separate
            hashtags. It is "original_tweets.p" in our case.
        :param y: None, needed for compatibility.
        :return: self.
        """
        self.graph = nx.Graph()

        x = self.drop_tweets_with_hashtags_that_occurred_less_than(
            x, self.minimal_hashtag_occurrence
        )

        hashtag_agg = defaultdict(list)

        self._hashtag_labels = set()
        self._users_labels = set()
        self._tweet_labels = set()

        if self.verbose:
            print("Building graph ...")
            tqdm.tqdm.pandas()
            x.progress_apply(
                lambda r: self._transform_single_row(hashtag_agg, r), axis=1
            )
        else:
            x.apply(
                lambda r: self._transform_single_row(hashtag_agg, r), axis=1
            )

        self._refine_matrix_with_additional_connections()

        self._hashtag_labels = np.asarray(list(sorted(self._hashtag_labels)))
        self._users_labels = np.asarray(list(sorted(self._users_labels)))
        self._tweet_labels = np.asarray(list(sorted(self._tweet_labels)))

        if self.verbose:
            print("Building incidence matrix ...")
        incidence_matrix = self._get_binary_incidence_matrix()[
            : len(self._hashtag_labels), len(self._hashtag_labels) :
        ]
        weighted_adjacency_matrix_of_tags = incidence_matrix.dot(
            incidence_matrix.T
        )
        weighted_adjacency_matrix_of_tags.setdiag(0)

        if self.verbose:
            print("Building hashtag graph ...")

        hashtag_graph = nx.from_scipy_sparse_matrix(
            weighted_adjacency_matrix_of_tags
        )

        weighted_degree = np.asarray(
            list(map(itemgetter(1), hashtag_graph.degree(weight="weight")))
        )
        matrix_weighted_degree = sps.diags([1 / (weighted_degree + 1e-8)], [0])
        self._transition_matrix = weighted_adjacency_matrix_of_tags.dot(
            matrix_weighted_degree
        )

        if self.verbose:
            print("Calculating tf idf ...")

        document_list = [
            " ".join(hashtag_agg[key]) for key in self._hashtag_labels
        ]

        # it has normalization inside, so no L2 is necessary
        self._hashtags_tf_idf_vectorizer = TfidfVectorizer(norm="l2")
        self._hashtags_tf_idf_representation = (
            self._hashtags_tf_idf_vectorizer.fit_transform( document_list )
        )

        if self.verbose:
            print("Fitting hashtag encoders ...")

        # [:-4] because each hashtag has "_tag" postfix to distinguish it in
        # the graph
        self._hashtag_encoder.fit([lab[:-4] for lab in self._hashtag_labels])

        return self

    def transform(
        self, x: Union[Tuple[Tuple[str, ...], ...], Tuple[str, ...]], **kwargs
    ) -> np.ndarray:
        """
        For a given tweet represented as a list of lemmas recommends hashtags.
        :param x: tuple of str or str. If tuple is str, strs are lemmas of the
            tweet. If single str, it is assumed that lemmatization has to be
            performed.
        :key query: Iterable[str]. Optional query hashtag for each tweet in
            `x`.
        :return: np.ndarray of recommended hashtags.
        """
        lemmatised = list(x[:])
        if isinstance(lemmatised[0], str):
            for i, xi in enumerate(lemmatised):
                lemmatised[i] = get_wcrft2_results_for_text(xi)
        if isinstance(lemmatised[0], list):
            for i, xi in enumerate(lemmatised):
                lemmatised[i] = " ".join(xi)

        query_hashtags = kwargs.get("query", None)
        if query_hashtags is not None:
            assert len(query_hashtags) == len(
                x
            ), "If at least 1 query is given, the array should have the " \
               "same dimension as input `x`"
        if isinstance(query_hashtags, str):
            query_hashtags = [query_hashtags] * len(lemmatised)

        # as in fit, vectorizer has normalization inside ...
        tf_idf_vectors = self._hashtags_tf_idf_vectorizer.transform(lemmatised)

        # ... so this simplifies to cosine similarity - no
        # normalisation required
        similarities = self._hashtags_tf_idf_representation.dot(
            tf_idf_vectors.T
        ).T.toarray()
        preference_vectors = self._get_preference_vectors(
            similarities, query_hashtags
        )
        similarity_rank_vertices = self._random_walk(preference_vectors)

        best_indices = np.argsort(
            -similarities * similarity_rank_vertices, axis=1
        )
        result = self._hashtag_labels[best_indices].tolist()
        return self.post_process_result(result)

    def _get_preference_vectors(
        self,
        tweet_content_similarities: np.ndarray,
        query_hashtags: Optional[Tuple[str]],
    ) -> sps.csr_matrix:
        """
        Creates sparse matrix of preference vectors for each of N samples to
        recommend which are used to initialize random walk algorithm. If a
        query hashtag for a particular tweet is given, then it is used to
        create preference vector. The most similar hashtag is used otherwise.

        :param tweet_content_similarities: np.ndarray. Similarities between
            given N samples to predict and already fit hashtags.
        :param query_hashtags: Optional collection of strings. Used to
            initialize preference vector in accordance with the paper.
        :return: Sparse matrix of N one hot vectors.
        """

        def _get_using_similarities(similarity_vector):
            query_hashtag_index = np.argmax(similarity_vector)
            vec = np.zeros((len(self._hashtag_labels),))
            vec[query_hashtag_index] = 1
            return vec

        preference_vectors = []
        for i in range(len(tweet_content_similarities)):
            if query_hashtags is None or query_hashtags[i] is None:
                preference_vector = _get_using_similarities(
                    tweet_content_similarities[i]
                )
            else:
                try:
                    preference_vector = np.asarray(
                        self._hashtag_encoder.transform([query_hashtags[i]])
                    )[0]
                except ValueError:
                    warnings.warn(
                        "Unknown hashtag: {}. Using the closest hashtag in "
                        "terms of content similarity".format(
                            query_hashtags[i]
                        )
                    )
                    preference_vector = _get_using_similarities(
                        tweet_content_similarities[i]
                    )
            preference_vectors.append(preference_vector)
        preference_vectors = np.vstack(preference_vectors)
        preference_vectors = sps.csr_matrix(
            preference_vectors, preference_vectors.shape, dtype=np.float32
        )

        return preference_vectors

    def _random_walk(self, preference_vectors: sps.csr_matrix) -> np.ndarray:
        """
        Performs random walk algorithm on graph using transition matrix
        calculated in `fit`, given similarities of input tweet to hashtags
        representations calculated as tf idf in `fit` method. Random walk
        lasts until no changes are noticed in node values or algorithm
        exceeded upper limit of possible iterations.

        :param preference_vectors: Sparse matrix N x M of length N, where N is
            a number of samples to recommend and M is a total number of
            hashtags used during fit.
        :return: Vector of length N, where each element consists probability
            of going from node representing hashtag in `preference_vector`
            to other nodes. The higher probability, the better hashtag.
        """
        similarity_rank_vertices = preference_vectors
        nb_iteration = 0
        while True:
            previous_similarity_rank_vertices = similarity_rank_vertices
            if self.verbose:
                print("Step: {}".format(nb_iteration + 1))

            similarity_rank_vertices = (
                self.damping_factor
                * similarity_rank_vertices.dot(self._transition_matrix)
                + (1 - self.damping_factor) * preference_vectors
            )

            diff = np.sum(
                np.abs(
                    similarity_rank_vertices
                    - previous_similarity_rank_vertices
                )
            )
            if (
                nb_iteration > 0
                and diff < self.minimal_random_walk_change_difference_value
            ):
                if self.verbose:
                    print("Converged with error: {:.6f}".format(diff))
                break

            nb_iteration += 1

            if nb_iteration > self.max_iterations:
                if self.verbose:
                    print(
                        "Random walk did not converge, current error: {:.6f}".format(
                            diff
                        )
                    )
                break
        return similarity_rank_vertices.toarray()

    @classmethod
    def post_process_result(cls, result: np.ndarray) -> np.ndarray:
        """
        Removes ending suffix that was used to distinguish between nodes with
        the same name but different category.

        :param result: list of strings. Result containing hashtags sorted by
            recommendation value
        :return: np.ndarray. Tags in nd.ndarray without suffix.
        """
        to_cut = len("_tag")
        return np.asarray(
            [
                [tag[:-to_cut] for tag in list_of_tags]
                for list_of_tags in result
            ]
        )
