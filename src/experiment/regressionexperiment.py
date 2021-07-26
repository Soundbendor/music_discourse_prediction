from .experiment import Experiment
from preprocessing.experimentset import ExperimentSet
from preprocessing.dataset import Dataset

from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from scipy.stats.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

N_SPLITS = 5

class RegressionExperiment(Experiment):
    
    def train_test_split(self, ds: Dataset, test_size: int, key: str):
        X, y = ds.get_data(key)
        return train_test_split(X, y, test_size=test_size)

    # TODO - get more stats and pretty-print them. Maybe output to a debug file? 

    def _cross_validate(self, estimator: Pipeline, expset: ExperimentSet):
        scores = []
        metrics = [pearsonr, spearmanr, mean_absolute_error]
        kfold = KFold(n_splits=N_SPLITS, shuffle=True).split(expset.X_train, expset.y_train)
        print("\n---Beginning cross validation---")
        for k, (i_train, i_test) in tqdm(enumerate(kfold), total=N_SPLITS):
            X_train, y_train = expset.X[i_train], expset.y[i_train]
            X_test, y_test = expset.X[i_test], expset.y[i_test]

            estimator.fit(X_train, y_train)
            yhat = estimator.predict(X_test)
            self.cv_stats(scores, metrics, y_test, yhat)
        self.display_stats(scores)

    def cv_stats(self, scores, metrics, y_true, y_pred):
        iter_score = {}
        for metric in metrics:
            iter_score[metric.__name__] = metric(y_true, y_pred)
        scores.append(iter_score)

    def display_stats(self, scores: list):
        for idx, score in enumerate(scores):
            print(f'\nIteration - {idx+1}')
            for k, v in score.items():
                print(f'Metric: {k} \t\t\t Score: {v}')

