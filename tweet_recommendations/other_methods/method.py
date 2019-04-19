from typing import *

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Method(TransformerMixin, BaseEstimator):
    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Method":
        raise NotImplementedError

    def transform(self, x: List[str]) -> List[str]:
        raise NotImplementedError

    @classmethod
    def drop_tweets_with_hashtags_that_occurred_less_than(cls, data: pd.DataFrame,
                                                          minimal_hashtag_occurrence: int) -> List[str]:
        hashtags = data["hashtags"].tolist()
        hashtags = [h['text'] for a_list in hashtags for h in a_list]
        counts = Counter(hashtags)
        filtered_tags = [t for t, count in counts.items() if
                         count >= minimal_hashtag_occurrence]
        return filtered_tags

    @classmethod
    def drop_tweets_without_given_hashtags(cls, data: pd.DataFrame, filtered_tags: List[str]) -> pd.DataFrame:
        data["hashtags"] = data["hashtags"].apply(
            lambda x: [elem for elem in x if elem["text"] in filtered_tags])
        data = data.drop(data[data["hashtags"].str.len() == 0].index)
        return data
