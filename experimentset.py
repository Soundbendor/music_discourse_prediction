from dataset import Dataset

class ExperimentSet:
    def __init__(self, ds: Dataset, key: str, split_method, test_size: int) -> None:
        self.key = key
        self.X, self.y = ds.get_data(key)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            split_method(ds, test_size, key)
