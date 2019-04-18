from typing import *

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Method(TransformerMixin, BaseEstimator):
    def fit(self, x: pd.DataFrame, y: Optional[pd.DataFrame] = None, **fit_params) -> "Method":
        raise NotImplementedError

    def transform(self, x: List[str]) -> List[str]:
        raise NotImplementedError
