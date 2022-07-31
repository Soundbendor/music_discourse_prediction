
import pandas as pd
import json

from .cvsummary import CVSummary
from .report import Report
from prediction.preprocessing.experimentset import ExperimentSet
from prediction.preprocessing.experimentfactory import ExperimentFactory
from prediction.preprocessing.dataset import Dataset

from scipy.stats.stats import pearsonr, spearmanr

from imblearn.base import BaseSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from abc import ABC, abstractmethod
from typing import Generator

from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

N_SPLITS = 5


class Experiment(ABC):
    def __init__(self, dataset: Dataset, config: ExperimentFactory, output) -> None:
        self.ds = dataset
        self.config = config
        self.metrics = [self.correl_pearson, self.correl_spearman]
        self.output_fname = output
        self.report = Report()

    def run_experiment(self) -> None:

        test_size = self.config.get_preprocessing_args()['test_size']
        sampler = self.config.get_sampling_strategy()
        model = self.config.get_model()
        fs = self.config.get_feature_selection_strategy()
        pipe = self._build_pipeline(fs, sampler, model)

        self.report.set_dataset_info(self.ds.summary)

        results_predicted = pd.DataFrame(columns=self._get_keys())
        results_actual = pd.DataFrame(columns=self._get_keys())

        # We'll just always grid search on valence, since it's the harder problem.
        gs_expset = ExperimentSet(self.ds, self._get_keys()[0], self.split_dataset, test_size)
        best_est = self._run_grid_search(pipe, gs_expset)

        with open(f"out/parameters/{self.output_fname}_params.json", 'a', encoding='utf-8') as param_file:
            json.dump({k: v for k, v in best_est.get_params().items() if 'model__' in k}, param_file)

        for key in self._get_keys():
            print(f'\nMaking predictions for {key}\n')

            expset = ExperimentSet(self.ds, key, self.split_dataset, test_size)

            y_pred = self._cross_validate(key, best_est, expset)
            results_predicted[key] = y_pred
            results_actual[key] = expset.y_test

        results_predicted.to_csv(f"out/predictions/{self.output_fname}_predictions_out.csv")
        self._generate_vis(results_predicted, results_actual)
        self.report.output_report(self.output_fname)

    def _build_pipeline(self, feature_selection, sampling_method: BaseSampler, model) -> ImbPipeline:
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
        print(gs_args['param_grid'])
        return GridSearchCV(
            estimator=estimator,
            param_grid=gs_args['param_grid'],
            refit=True,
            cv=gs_args['cv'],
            scoring=gs_args['scoring'],
            n_jobs=-1,
            verbose=2,
        )

    def _cross_validate(self, key: str, estimator, expset: ExperimentSet):
        cv_summary = CVSummary(self.metrics)
        # kfold = self._get_k_fold(N_SPLITS, expset)
        # print("\n---Beginning cross validation---")
        # for k, (i_train, i_test) in tqdm(enumerate(kfold), total=N_SPLITS):
        #     X_train, y_train = expset.X_train[i_train], expset.y_train[i_train]
        #     X_test, y_test = expset.X_train[i_test], expset.y_train[i_test]

        #     estimator.fit(X_train, y_train)
        #     y_hat = estimator.predict(X_test)
        #     cv_summary.score_cv(y_test, y_hat)

        estimator.fit(expset.X_train, expset.y_train)
        y_test_pred = estimator.predict(expset.X_test)
        cv_summary.score_cv(expset.y_test, y_test_pred)
        self.report.set_summary_stats(key, cv_summary)
        return y_test_pred

    def correl_pearson(self, y_test, y_hat):
        correl, _ = pearsonr(y_test, y_hat)
        return correl

    def correl_spearman(self, y_test, y_hat):
        correl, _ = spearmanr(y_test, y_hat)
        return correl

    @abstractmethod
    def split_dataset(self, ds: Dataset, test_size: int, key: str):
        pass

    @abstractmethod
    def _get_k_fold(self, n_splits: int, expset: ExperimentSet) -> Generator:
        pass

    # Defines which keys we will be running prediction for.
    @abstractmethod
    def _get_keys(self) -> list:
        pass

    @abstractmethod
    def _generate_vis(self, df_pred: pd.DataFrame, df_results: pd.DataFrame) -> None:
        pass


class ExperimentTypeNotFoundError(Exception):
    pass
