from collections import Counter
from itertools import chain

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.decomposition import TruncatedSVD

from tweet_recommendations.embeddings.word2vec import load_w2v_model
from tweet_recommendations.estimator import Estimator


class SIFEmbedding(Estimator):

    def __init__(self, w2v_model_path: str):
        self.w2v_keyed_vector = load_w2v_model(w2v_model_path)
        self.words_weights = None
        self._pc = None

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None,
            **fit_params) -> "SIFEmbedding":
        """

        :param x: DataFrame which should containt tweet text lemmas
        :param y: None for compatibility
        :param fit_params: min_word_occurences: int. Parameter limiting/filtering infrequent words
                           smoothing: int. Smoothing value parameter.
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
            word: smoothing / (smoothing + (count / total_words_count))
            for word, count
            in word_counts.items()}

        emb = self._get_weighted_average(x["lemmas"].tolist())

        self._pc = self._compute_pc(emb)

        return self

    def transform(self, x: List[str]) -> np.ndarray:
        """
        :param x: List of lemmas in a tweet.
        :return: Sentence embedding
        """
        x = List[x]
        sentence_embedding = self._get_sif_embedding(x)

        return sentence_embedding

    def fit_transform(self, x: pd.DataFrame, y=None, **fit_params) -> np.ndarray:
        self.fit(x, y, **fit_params)
        embeddings = self._get_sif_embedding(x["lemmas"].tolist())
        return embeddings

    def _get_weighted_average(self, sentences: List[List[str]]) -> np.ndarray:
        """
        Compute the weighted average vectors for given tweet content lemmas.
        :param sentences: List of tweet content lemmas list.
        :return: Weighted average of tweet embeddings.
        """
        n_samples = len(sentences)
        emb = np.zeros((n_samples, self.w2v_keyed_vector.vector_size))
        for i in range(n_samples):
            if sentences[i]:
                words_embeddings = []
                words_weights = []
                for word in sentences[i]:
                    words_embeddings.append(self.get_word_vector(word.lower()))
                    words_weights.append(self.words_weights.get(word.lower(), 0))
                sentence_word_embeddings = np.array(words_embeddings)
                sentence_word_weights = np.array(words_weights)

                emb[i, :] = sentence_word_weights.dot(sentence_word_embeddings) / len(
                    sentence_word_weights)
        return emb

    def _compute_pc(self, tweets_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute the first principal components for given tweet embedding matrix.
        :param tweets_embeddings: array of tweet embeddings.
        :return: svd.component_: array of first pca principal component
        for each tweet embedding.
        """
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
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
        :param We: We[i,:] is the vector for word i
        :return: emb, emb[i, :] is the embedding for sentence i
        """
        emb = self._get_weighted_average(sentences)
        emb = self._remove_pc(emb)
        return emb

    def get_word_vector(self, word: str) -> np.ndarray:
        try:
            word_embedding = self.w2v_keyed_vector[word.lower()]
        except Exception:
            print(
                "Key '{}' not found in w2v keyed vector. Returning zeros.".format(word))
            word_embedding = np.zeros(self.w2v_keyed_vector.vector_size)
        return word_embedding
