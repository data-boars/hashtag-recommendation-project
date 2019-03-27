from collections import defaultdict, Counter
from operator import itemgetter
from typing import *

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sps
import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

from tweet_recommendations.utils.clients import get_wcrft2_results_for_text
from .method import Method


class GraphSummarizationMethod(Method):
    def __init__(self, minimal_random_walk_change_difference_value: float,
                 damping_factor: float,
                 max_iterations: int,
                 verbose: bool = False):
        """
        Creates estimator for predicting hashtag based on graph construction
        from https://sci-hub.tw/https://ieeexplore.ieee.org/document/7300890
        :param minimal_random_walk_change_difference_value: float. Small value.
            It is used for stop criterion in random walk. If difference between random walk values from previous
            step and random walk values from current step is less than that value, then algorithm is stopped.
        :param damping_factor: float. 0 <= damping_factor <= 1. Probability between random walk and restart.
        :param max_iterations: int. max_iterations > 0. Maximal value of iterations to perform for random walk.
        :param verbose: bool. Whether method should be talkative.
        """
        self.graph: nx.Graph = None

        self._hashtags_tf_idf_vectorizer: TfidfVectorizer = None
        self._hashtags_tf_idf_representation: np.ndarray = None

        self._hashtag_labels: Union[set, np.ndarray] = None
        self._users_labels: Union[set, np.ndarray] = None
        self._tweet_labels: Union[set, np.ndarray] = None

        self._transition_matrix: np.ndarray = None

        self.minimal_random_walk_change_difference_value = minimal_random_walk_change_difference_value
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.verbose = verbose

    def _transform_single_row(self, hashtag_agg: Dict, row: pd.Series):
        """
        Transforms single row of pandas `original_tweets_with_lemmas.p` to graph. Suffixes in node names are needed due
        to intersection between hashtags and user names.
        :param hashtag_agg: Dict[str, List[str]]. Agreggation for hashtag tf-idf calculation. Each hashtag is
            represented by list of words which it occurs in.
        :param row: pd.Series. Single row of aforementioned dataframe.
        :return: None.
        """
        user_name = row["username"] + "_user"
        tweet_id = row["id_str"] + "_tweet"
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

    def _refine_matrix_with_additional_connections(self, verbose: bool):
        """
        Adds edges between hashtag nodes if they share the same user.
        :param verbose: bool. Print progressbar while iterating through nodes.
        :return: None.
        """
        new_graph = self.graph.copy()
        for node in tqdm.tqdm(self.graph.nodes(), disable=not verbose):
            if self.graph.node[node]["type"] == "hashtag":
                for neighbour in self.graph.neighbors(node):
                    if self.graph.node[neighbour]["type"] == "username":
                        for other_node in self.graph.neighbors(neighbour):
                            if self.graph.node[other_node]["type"] == "hashtag" and not self.graph.has_edge(node,
                                                                                                            other_node):
                                new_graph.add_edge(node, other_node)
        self.graph = new_graph

    def _get_binary_incidence_matrix(self):
        # needed to preserve ordering
        all_labels = np.concatenate((self._hashtag_labels, self._tweet_labels, self._users_labels))
        incidence_matrix = nx.adjacency_matrix(self.graph, nodelist=all_labels)

        return incidence_matrix

    def _drop_hashtag_that_occurred_less_than(self, data: pd.DataFrame,
                                              minimal_hashtag_occurrence: int) -> pd.DataFrame:
        hashtags = data["hashtags"].tolist()
        hashtags = [h['text'] for a_list in hashtags for h in a_list]
        counts = Counter(hashtags)
        filtered_tags = [t for t, count in counts.items() if count >= minimal_hashtag_occurrence]
        data["hashtags"] = data["hashtags"].apply(lambda x: [elem for elem in x if elem["text"] in filtered_tags])
        data = data[data["hashtags"].str.len() > 0]
        return data

    def fit(self, x: pd.DataFrame, y=None, **fit_params) -> "Method":
        """
        Builds tri partite graph of Users - Hashtags - Tweets. Hashtags are connected if has the same user.
        :param x: pd.DataFrame with tweet content, user id, and separate hashtags. It is "original_tweets.p" in our
            case.
        :param y: None, needed for compatibility.
        :param fit_params:
            minimal_hashtag_occurrence: int. If hashtag occurred less than this number then it's not considered during
                prediction (simply removed). To include all hashtags put number <= 0.
        :return: self.
        """
        self.graph = nx.Graph()
        minimal_hashtag_occurence = fit_params["minimal_hashtag_occurence"]

        x = self._drop_hashtag_that_occurred_less_than(x, minimal_hashtag_occurence)

        hashtag_agg = defaultdict(list)

        self._hashtag_labels = set()
        self._users_labels = set()
        self._tweet_labels = set()

        if self.verbose:
            print("Building graph ...")
            tqdm.tqdm.pandas()
            x.progress_apply(lambda r: self._transform_single_row(hashtag_agg, r), axis=1)
        else:
            x.apply(lambda r: self._transform_single_row(hashtag_agg, r), axis=1)

        self._refine_matrix_with_additional_connections(self.verbose)

        self._hashtag_labels = np.asarray(list(self._hashtag_labels))
        self._users_labels = np.asarray(list(self._users_labels))
        self._tweet_labels = np.asarray(list(self._tweet_labels))

        if self.verbose:
            print("Building incidence matrix ...")
        incidence_matrix = self._get_binary_incidence_matrix()[:len(self._hashtag_labels), len(self._hashtag_labels):]
        weighted_adjacency_matrix_of_tags = incidence_matrix.dot(incidence_matrix.T)
        weighted_adjacency_matrix_of_tags.setdiag(0)

        if self.verbose:
            print("Building hashtag graph ...")

        hashtag_graph = nx.from_scipy_sparse_matrix(weighted_adjacency_matrix_of_tags)

        weighted_degree = np.asarray(list(map(itemgetter(1), hashtag_graph.degree(weight="weight"))))
        matrix_weighted_degree = sps.diags([1 / (weighted_degree + 1e-8)], [0])
        self._transition_matrix = weighted_adjacency_matrix_of_tags.dot(matrix_weighted_degree)

        if self.verbose:
            print("Calculating tf idf ...")

        document_list = [' '.join(hashtag_agg[key]) for key in self._hashtag_labels]

        # it has normalization inside, so no L2 is necessary
        self._hashtags_tf_idf_vectorizer = TfidfVectorizer(norm="l2")
        self._hashtags_tf_idf_representation = self._hashtags_tf_idf_vectorizer.fit_transform(document_list)

        return self

    def transform(self, x: Union[List[str], str]) -> List[str]:
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

        # as in fit, vectorizer has normalization inside ...
        tf_idf_vector = self._hashtags_tf_idf_vectorizer.transform([x])

        # ... so this simplifies to cosine similarity
        similarities = self._hashtags_tf_idf_representation.dot(tf_idf_vector.T).toarray()[:, 0]
        query_hashtag_index = np.argmax(similarities)

        preference_vector = np.zeros((len(self._hashtag_labels, )))[..., np.newaxis]
        preference_vector[query_hashtag_index] = 1
        preference_vector = sps.csr_matrix(preference_vector, shape=preference_vector.shape, dtype=np.float32)

        similarity_rank_vertices = preference_vector
        previous_similarity_rank_vertices: np.ndarray = preference_vector
        nb_iteration = 0

        while True:
            if self.verbose:
                print("Step: {}".format(nb_iteration + 1))

            similarity_rank_vertices = self.damping_factor * self._transition_matrix.dot(similarity_rank_vertices) + (
                    1 - self.damping_factor) * preference_vector

            diff = np.abs(np.sum(similarity_rank_vertices - previous_similarity_rank_vertices))
            if nb_iteration > 0 and diff < self.minimal_random_walk_change_difference_value:
                if self.verbose:
                    print("Converged with error: {:.6f}".format(diff))
                break

            previous_similarity_rank_vertices = similarity_rank_vertices
            nb_iteration += 1

            if nb_iteration > self.max_iterations:
                if self.verbose:
                    print("Random walk did not converged, current error: {:.6f}".format(diff))
                break

        similarity_rank_vertices = np.squeeze(similarity_rank_vertices.toarray())

        best_indices = np.argsort(-similarities * similarity_rank_vertices)
        result = self._hashtag_labels[best_indices].tolist()
        return self.post_process_result(result)

    def post_process_result(self, result: List[str]):
        """
        Removes ending suffix that was used to distinguish between nodes with the same name but different category.
        :param result: list of strings. Result containing hashtags sorted by recommendation value
        :return: list of strings. Tags in nd.ndarray without suffix.
        """
        to_cut = len("_tag")
        return [tag[:-to_cut] for tag in result]
