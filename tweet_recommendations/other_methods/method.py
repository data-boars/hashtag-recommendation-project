from typing import *

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class Method(TransformerMixin, BaseEstimator):
    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Method":
        raise NotImplementedError

    def transform(self, x: List[str], **transform_params) -> List[str]:
        raise NotImplementedError
