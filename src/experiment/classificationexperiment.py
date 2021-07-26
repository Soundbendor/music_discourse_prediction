from .experiment import Experiment
from preprocessing.experimentset import ExperimentSet
from preprocessing.experimentfactory import ExperimentFactory
from preprocessing.dataset import ClassificationDataset, Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score


class_names = {
    'happy': 0,
    'upset': 1,
    'depressed': 2,
    'calm': 3 
}


class ClassificationExperiment(Experiment):

    def __init__(self, dataset: Dataset, config: ExperimentFactory) -> None:
        super().__init__(dataset, config)
        self.metrics.append(f1_score)
        self.ds = self._label_data(self.ds)
    
    def split_dataset(self, ds: Dataset, test_size: int, key: str):
        X, y = ds.get_data(key)
        return train_test_split(X, y, stratify=ds.y, test_size=test_size)

    def _get_k_fold(self, n_splits: int, expset: ExperimentSet):
        return StratifiedKFold(n_splits, shuffle=True).split(expset.X_train, expset.y_train)

    def _label_data(self, ds: Dataset) -> ClassificationDataset: 

        c_ds = ClassificationDataset(ds)

        c_ds.df.loc[(ds.df['existing_valence'] >= 0) & (c_ds.df['existing_arousal'] >= 0), 'class'] = class_names['happy']
        c_ds.df.loc[(ds.df['existing_valence'] >= 0) & (c_ds.df['existing_arousal'] < 0), 'class'] = class_names['upset']
        c_ds.df.loc[(ds.df['existing_valence'] < 0) & (c_ds.df['existing_arousal'] < 0), 'class'] = class_names['depressed']
        c_ds.df.loc[(ds.df['existing_valence'] < 0) & (c_ds.df['existing_arousal'] >= 0), 'class'] = class_names['calm']

        return c_ds

