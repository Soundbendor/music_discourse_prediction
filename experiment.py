from experimentfactory import ExperimentFactory
from dataset import Dataset

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod


class Experiment(ABC):
    def __init__(self, dataset: Dataset, config: ExperimentFactory) -> None:
        self.ds = dataset
        self.config = config


    def run_experiment(self):
        self.X_train, self.X_test, \
            self.y_train, self.y_test = \
            self.train_test_split(self.ds, self.config.get_preprocessing_args()['test_size'])

        sampler = self.config.get_sampling_strategy()
        model = self.config.get_model()
        fs = self.config.get_feature_selection_strategy()
        
        print(self.y_train)

        pipe = self._build_pipeline(fs, sampler, model)
        
        for key in self.config.get_y_keys():
            print(f'\nMaking predictions for {key}\n')
            best_est = self._run_grid_search(pipe, key)
            self._cross_validate(best_est)
        

    def _build_pipeline(self, feature_selection, sampling_method, model):
        return ImbPipeline([
            ('standardscaler', StandardScaler()),
            ('featureselection', feature_selection),
            ('sampler', sampling_method),
            ('model', model)
        ])

    def _run_grid_search(self, estimator, key: str):
        gs_args = self.config.get_grid_search()
        if(not gs_args['grid_search']):
            return estimator

        gs = self._build_grid_search(estimator, gs_args)
        gs = gs.fit(self.X_train, self.y_train[key])
        return gs.best_estimator_


    def _build_grid_search(self, estimator, gs_args: dict):

        return GridSearchCV(
            estimator= estimator, 
            param_grid= gs_args['param_grid'],
            refit= True,
            cv= gs_args['cv'],
            scoring= gs_args['scoring'],
            n_jobs= -1,
            verbose=2
        )

    @abstractmethod
    def train_test_split(self, ds: Dataset, test_size: int):
        pass

    @abstractmethod
    def _cross_validate(self, estimator):
        pass

    

class ExperimentTypeNotFoundError(Exception):
    pass


# bulid an estimator
# build a grid search with that estimator (if relevant)
# run the grid search to return the best parameters for that param grid
# build a cross validator with the pipeline created
# run cross validation
# create final predictions
# output result graphics