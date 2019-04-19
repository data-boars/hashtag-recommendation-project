from collections import Counter
from itertools import chain
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import tqdm
from gensim.models import KeyedVectors
from sklearn.decomposition import TruncatedSVD

from tweet_recommendations.other_methods.method import Method
from tweet_recommendations.utils.clients import get_wcrft2_results_for_text


class SIFEmbedding(Method):

    def __init__(self, path_to_keyedvectors_model: str, verbose: bool = False):
        """
        Method for creating embeddings out of words and combining them into single tweet embedding with modified
        weighted average of those embeddings.
        :param path_to_keyedvectors_model: Path to converted by script `convert_embedding_model_to_mmap.py` gensim
            model. It can either word2vec or fasttext, `gensim` handles both.
        :param verbose: Whether method should be verbose.
        """
        self.verbose = verbose

        if self.verbose:
            print("Loading keyed vectors model ...")
        self.keyed_vector_model = KeyedVectors.load(path_to_keyedvectors_model, mmap="r")
        self.words_weights = None
        self._pc = None

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None,
            **fit_params) -> "SIFEmbedding":
        """

        :param x: DataFrame which should contain tweet text lemmas
        :param y: None for compatibility
        :param fit_params: min_word_occurrences: int. Parameter limiting/filtering infrequent words.
                           smoothing: int. Smoothing value parameter.
                           random_state: int. Parameter used for calculating principal components.

        :return: self
        """
        min_word_occurence = fit_params.get("min_word_occurences", 0)
        smoothing = fit_params.get("smoothing", 1)

        all_words_in_dataset = list(chain.from_iterable(x["lemmas"].values))
        word_counts = dict(Counter(all_words_in_dataset))
        if min_word_occurence > 0:
            word_counts = {lemma: count for lemma, count in
                           word_counts.items() if count >= min_word_occurence}

        total_words_count = len(all_words_in_dataset)
        self.words_weights = {
            word.lower(): smoothing / (smoothing + (count / total_words_count))
            for word, count
            in word_counts.items()}

        emb = self._get_weighted_average_embeddings(x["lemmas"].values)

        self._pc = self._compute_pc(emb, fit_params.get("random_state", None))

        return self

    def transform(self, x: Union[List[List[str]], List[str]], **kwargs) -> np.ndarray:
        """
        For a given tweet represented as a list of lemmas recommends hashtags.
        :param x: list of list of str or list of str. If first argument of x is a list is str, it is assumed that list
            contains already lemmatized text. If single str is present as first element, it is assumed
            that lemmatization has to be performed.
        :return: np.ndarray of sentences embeddings.
        """
        lemmatized = []
        if isinstance(x, pd.DataFrame):
            lemmatized = x["lemmas"].to_list()
        elif isinstance(x[0], str):
            for i, xi in enumerate(x):
                lemmatized.append(get_wcrft2_results_for_text(xi))
        elif isinstance(x[0], list):
            for i, xi in enumerate(x):
                lemmatized.append(xi)
        output = self._get_sif_embedding(lemmatized)
        return np.asarray(output)

    def _get_weighted_average_embeddings(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Compute the weighted average vectors for given tweet content lemmas.
        :param sentences: List of tweet content lemmas list.
        :return: Weighted average of tweet embeddings.
        """
        n_samples = len(sentences)
        emb = np.zeros((n_samples, self.keyed_vector_model.vector_size))
        for i in tqdm.trange(n_samples, disable=not self.verbose):
            if sentences[i]:
                words_embeddings = []
                words_weights = []
                for word in sentences[i]:
                    try:
                        word_embedding = self.keyed_vector_model.word_vec(word.lower())
                        words_embeddings.append(word_embedding)
                        words_weights.append(self.words_weights.get(word.lower(), 0))
                    except KeyError:
                        continue
                sentence_word_embeddings = np.array(words_embeddings)
                sentence_word_weights = np.array(words_weights)

                if len(sentence_word_weights) > 0:
                    emb[i, :] = sentence_word_weights.dot(sentence_word_embeddings) / len(
                        sentence_word_weights)
        return emb

    def _compute_pc(self, tweets_embeddings: np.ndarray, random_state) -> np.ndarray:
        """
        Compute the first principal components for given tweet embedding matrix.
        :param tweets_embeddings: array of tweet embeddings.
        :return: svd.component_: array of first pca principal component
        for each tweet embedding.
        """
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=random_state)
        svd.fit(tweets_embeddings)
        return svd.components_

    def _remove_pc(self, tweets_embeddings: np.ndarray) -> np.ndarray:
        """
        Remove the projection on the principal components for each tweet embedding.
        :param tweets_embeddings: array of tweet embeddings.
        :return: tweets embeddings after removing its projection.
        """
        tweet_embeddings_with_first_pc_removed = tweets_embeddings - tweets_embeddings.dot(
            self._pc.transpose()) * self._pc
        return tweet_embeddings_with_first_pc_removed

    def _get_sif_embedding(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Compute tweet content embedding using weighted average with
        removing the projection on the first principal component
        :param sentences: sentences is a list of lemmatized tweet word list
        :return: emb is a array of tweets embeddings
        """
        emb = self._get_weighted_average_embeddings(sentences)
        emb = self._remove_pc(emb)
        return emb
