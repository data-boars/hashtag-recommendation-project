import numpy as np
from sklearn.preprocessing import LabelEncoder


class ModifiedOneHotEncoder(LabelEncoder):
    """
    Class needed because there are no methods in sklearn that would be suitable for encoding words as one hots. Tried
    combining `LabelEncoder` with `OneHotEncoder` without success, so this class was created.
    """
    def __init__(self):
        self._max_class_num = -1

    def fit(self, y):
        encoded = super().fit_transform(y)
        self._max_class_num = np.max(encoded)
        return self

    def transform(self, y):
        encoded = np.asarray(super().transform(y))
        one_hot_matrix = np.zeros((len(encoded), self._max_class_num))
        one_hot_matrix[:, y] = 1
        return one_hot_matrix
