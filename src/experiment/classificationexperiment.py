from preprocessing.experimentset import ExperimentSet
from preprocessing.dataset import ClassificationDataset, Dataset
from .experiment import Experiment

from sklearn.model_selection import train_test_split

'''
WARNING - This class is not ready yet!
The y-label in dataset needs to be changed to be classes, not regression values!
'''

class_names = {
    'happy': 0,
    'upset': 1,
    'depressed': 2,
    'calm': 3 
}


class ClassificationExperiment(Experiment):

    def run_experiment(self):

        test_size = self.config.get_preprocessing_args()['test_size']
        sampler = self.config.get_sampling_strategy()
        model = self.config.get_model()
        fs = self.config.get_feature_selection_strategy()
        pipe = self._build_pipeline(fs, sampler, model)

        for key in self.config.get_y_keys():
            print(f'\nMaking classification predictions for {key}\n')
            
            c_ds = self._label_data(self.ds)

            expset = ExperimentSet(c_ds, key, self.train_test_split, test_size)

            best_est = self._run_grid_search(pipe, expset)
            self._cross_validate(best_est, expset)
    
    def train_test_split(self, ds: Dataset, test_size: int, key: str):
        X, y = ds.get_data(key)
        return train_test_split(X, y, stratify=ds.y, test_size=test_size)

    def _label_data(self, ds: Dataset) -> ClassificationDataset: 

        c_ds = ClassificationDataset(ds)

        c_ds.df.loc[(ds.df['existing_valence'] >= 0) & (c_ds.df['existing_arousal'] >= 0), 'class'] = class_names['happy']
        c_ds.df.loc[(ds.df['existing_valence'] >= 0) & (c_ds.df['existing_arousal'] < 0), 'class'] = class_names['upset']
        c_ds.df.loc[(ds.df['existing_valence'] < 0) & (c_ds.df['existing_arousal'] < 0), 'class'] = class_names['depressed']
        c_ds.df.loc[(ds.df['existing_valence'] < 0) & (c_ds.df['existing_arousal'] >= 0), 'class'] = class_names['calm']

        return c_ds


    # TODO - Cross-validation. Similar to reg-exp's cv, but use StratifiedKFold
