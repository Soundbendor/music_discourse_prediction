from numpy.core.fromnumeric import mean
from experiment import Experiment
from dataset import Dataset

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

    def _cross_validate(self, estimator):
        scores = []
        metrics = [pearsonr, spearmanr, mean_absolute_error]
        kfold = KFold(n_splits=N_SPLITS, shuffle=True).split(self.X_train, self.y_train)
        print("\n---Beginning cross validation---")
        for k, (i_train, i_test) in tqdm(enumerate(kfold), total=N_SPLITS):
            X_train, y_train = self.ds.X.iloc[i_train], self.ds.y.iloc[i_train]
            X_test, y_test = self.ds.X.iloc[i_test], self.ds.y.iloc[i_test]

            estimator.fit(X_train, y_train)
            yhat = estimator.predict(X_test)
            self.cv_stats(scores, metrics, y_test, yhat)
        print(scores)

    def cv_stats(self, scores, metrics, y_true, y_pred):
        iter_score = {}
        for metric in metrics:
            print(type(y_true))
            iter_score[metric.__name__] = metric(y_true, y_pred)
        scores.append(iter_score)

        

