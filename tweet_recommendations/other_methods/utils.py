import numpy as np
from sklearn.preprocessing import LabelEncoder


class ModifiedOneHotEncoder(LabelEncoder):
    """
    Class needed because there are no methods in sklearn that would be suitable for encoding words as one hots. Tried
    combining `LabelEncoder` with `OneHotEncoder` without success, so this class was created.
    """
    def __init__(self):
        self._coding_book: np.ndarray = None

    def fit(self, y):
        encoding = np.asarray(super().fit_transform(y))
        self._coding_book = np.eye(np.max(encoding) + 1)
        return self

    def transform(self, y):
        encoded = np.asarray(super().transform(y))
        one_hot = self._coding_book[encoded]
        return one_hot
