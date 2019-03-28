from typing import List, Optional

import pandas as pd

from tweet_recommendations.estimator import Estimator
from .fast_text import load_fasttext_model
from .word2vec import load_w2v_model


class FastText(Estimator):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_fasttext_model(model_path)

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Estimator":
        return self

    def transform(self, x: List[str]) -> List[float]:
        all_embeddings = []
        for word in x:
            all_embeddings.append(self.model.get_word_vector(word))
        return all_embeddings


class Word2Vec(Estimator):
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = load_w2v_model(model_path)

    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Estimator":
        return self

    def transform(self, x: List[str]) -> List[float]:
        all_embeddings = []
        for word in x:
            try:
                word_embedding = self.model[word.lower()]
                all_embeddings.append(word_embedding)
            except KeyError:
                print("Key '{}' not found in w2v dict".format(word))
        return all_embeddings
