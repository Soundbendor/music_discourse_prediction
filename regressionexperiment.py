from experiment import Experiment
from dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from tqdm import tqdm

N_SPLITS = 5

class RegressionExperiment(Experiment):
    
    def train_test_split(self, ds: Dataset, test_size: int):
        return train_test_split(ds.X, ds.y, test_size=test_size)

    # TODO - get more stats and pretty-print them. Maybe output to a debug file? 

    def _cross_validate(self, estimator):
        scores = []
        kfold = KFold(n_splits=N_SPLITS, shuffle=True).split(self.X_train, self.y_train)
        print("\n---Beginning cross validation---")
        for k, (i_train, i_test) in tqdm(enumerate(kfold), total=N_SPLITS):
            X_train, y_train = self.ds.X.iloc[i_train], self.ds.y.iloc[i_train]
            X_val, y_val = self.ds.X.iloc[i_test], self.ds.y.iloc[i_test]

            estimator.fit(X_train, y_train)
            yhat = estimator.predict(X_val)
            scores.append(estimator.score(X_val, y_val))

        print(scores)

