from numpy import average
from .experiment import Experiment
from preprocessing.experimentset import ExperimentSet
from preprocessing.experimentfactory import ExperimentFactory
from preprocessing.dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score




class ClassificationExperiment(Experiment):

    def __init__(self, dataset: Dataset, config: ExperimentFactory, output) -> None:
        super().__init__(dataset, config, output)
        self.metrics.append(self.weighted_f1)

    def weighted_f1(self, y_true, y_pred):
        return f1_score(y_true, y_pred, average= 'weighted')
    
    def split_dataset(self, ds: Dataset, test_size: int, key: str):
        X, y = ds.get_data(key)
        return train_test_split(X, y, stratify=y, test_size=test_size)

    def _get_k_fold(self, n_splits: int, expset: ExperimentSet):
        return StratifiedKFold(n_splits, shuffle=True).split(expset.X_train, expset.y_train)

    def _get_keys(self) -> list: 
        return [self.ds.label_key]


