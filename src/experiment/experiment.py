from sklearn.pipeline import Pipeline
from preprocessing.experimentset import ExperimentSet
from preprocessing.experimentfactory import ExperimentFactory
from preprocessing.dataset import Dataset

from scipy.stats.stats import pearsonr, spearmanr

from imblearn.pipeline import Pipeline as ImbPipeline

from abc import ABC, abstractmethod

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

N_SPLITS = 5

class Experiment(ABC):
    def __init__(self, dataset: Dataset, config: ExperimentFactory, output) -> None:
        self.ds = dataset
        self.config = config
        self.metrics = [pearsonr, spearmanr]
        self.output = output


    def run_experiment(self):

        test_size = self.config.get_preprocessing_args()['test_size']
        sampler = self.config.get_sampling_strategy()
        model = self.config.get_model()
        fs = self.config.get_feature_selection_strategy()
        pipe = self._build_pipeline(fs, sampler, model)
        
        for key in self._get_keys():
            print(f'\nMaking predictions for {key}\n')

            expset = ExperimentSet(self.ds, key, self.split_dataset, test_size)

            best_est = self._run_grid_search(pipe, expset)
            self._cross_validate(best_est, expset)
      

    def _build_pipeline(self, feature_selection, sampling_method, model):
        return ImbPipeline([
            ('standardscaler', StandardScaler()),
            ('featureselection', feature_selection),
            ('sampler', sampling_method),
            ('model', model)
        ])

    def _run_grid_search(self, estimator, expset: ExperimentSet):
        gs_args = self.config.get_grid_search()
        if(not gs_args['grid_search']):
            return estimator

        gs = self._build_grid_search(estimator, gs_args)
        gs = gs.fit(expset.X_train, expset.y_train)
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

    def _cross_validate(self, estimator: Pipeline, expset: ExperimentSet):
        scores = []
        kfold = self._get_k_fold(N_SPLITS, expset)
        print("\n---Beginning cross validation---")
        for k, (i_train, i_test) in tqdm(enumerate(kfold), total=N_SPLITS):
            X_train, y_train = expset.X[i_train], expset.y[i_train]
            X_test, y_test = expset.X[i_test], expset.y[i_test]

            estimator.fit(X_train, y_train)
            yhat = estimator.predict(X_test)
            self.cv_stats(scores, y_test, yhat)
        self.display_stats(scores)

    def cv_stats(self, scores, y_true, y_pred):
        iter_score = {}
        for metric in self.metrics:
            iter_score[metric.__name__] = metric(y_true, y_pred)
        scores.append(iter_score)

    def display_stats(self, scores: list):
        for idx, score in enumerate(scores):
            print(f'\nIteration - {idx+1}')
            for k, v in score.items():
                print(f'Metric: {k} \t\t\t Score: {v}')


    @abstractmethod
    def split_dataset(self, ds: Dataset, test_size: int, key: str):
        pass

    @abstractmethod
    def _get_k_fold(self, n_splits: int, expset: ExperimentSet):
        pass

    # Defines which keys we will be running prediction for.
    @abstractmethod
    def _get_keys(self) -> list: 
        pass

    

class ExperimentTypeNotFoundError(Exception):
    pass

