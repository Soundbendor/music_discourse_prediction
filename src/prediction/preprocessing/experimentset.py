from .dataset import Dataset
from sklearn.preprocessing import MinMaxScaler


class ExperimentSet:
    def __init__(self, ds: Dataset, key: str, split_method, test_size: int) -> None:
        self.key = key
        self.X, self.y = ds.get_data(key)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            split_method(ds, test_size, key)
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # self.y_test = scaler.fit_transform(self.y_test.reshape(-1, 1)).ravel()
        # self.y_train = scaler.fit_transform(self.y_train.reshape(-1, 1)).ravel()


