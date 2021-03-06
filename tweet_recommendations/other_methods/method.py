from collections import Counter
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Method(TransformerMixin, BaseEstimator):
    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Method":
        raise NotImplementedError

    def transform(self, x: Union[Tuple[Tuple[str, ...], ...], Tuple[str, ...]], **transform_params) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def drop_tweets_with_hashtags_that_occurred_less_than(cls, data: pd.DataFrame,
                                                          minimal_hashtag_occurrence: int) -> pd.DataFrame:
        hashtags = data["hashtags"].tolist()
        hashtags = [h['text'] for a_list in hashtags for h in a_list]
        counts = Counter(hashtags)
        filtered_tags = {t for t, count in counts.items() if
                         count >= minimal_hashtag_occurrence}
        data["hashtags"] = data["hashtags"].apply(
            lambda x: [elem for elem in x if elem["text"] in filtered_tags])

        data = data.drop(data[data["hashtags"].str.len() == 0].index).reset_index()
        return data
