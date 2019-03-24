import os

import pandas as pd
from sklearn.base import TransformerMixin


class TaggerToygerLemmatizer(TransformerMixin):
    def __init__(self, path: str, verbose: bool = False):
        """
        Creates lemmatizer based on Tagger Toyger tool
        :param path: Path `.p` file containing dataframe with tweet id and lemmas for this tweet.
        """
        self.path = path
        self.output_path = os.path.join(
            os.path.dirname(self.path), os.pardir, "source_data", "original_tweets_with_lemmas.p"
        )
        self.verbose = verbose

    def fit(self, x=None, y=None, **kwargs):
        """
        Does nothing, needed for sklearn pipeline only.
        :param x: For compatibility only.
        :param y: For compatibility only.
        :return: self.
        """
        return self

    def transform(self, x: pd.DataFrame):
        """
        Adds new column base on tweets content in `original_tweets.p` with lemmas from tagger toyger. New column name:
        "lemmas".
        :param x: pd.DataFrame loaded from `original_tweets.p`.
        :return: pd.DataFrame with new column "lemmas".
        """
        if self.verbose:
            print("Constructing lemmas ...")

        if os.path.exists(self.output_path):
            return pd.read_pickle(self.output_path)
        lemmas = pd.read_pickle(self.path)
        joined = x.merge(lemmas, on="id_str", how="inner", suffixes=('', '_temp'))
        joined.to_pickle(self.output_path)

        return joined
