from experiment import Experiment
from dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


class RegressionExperiment(Experiment):
    
    def train_test_split(self, ds: Dataset, test_size: int):
        return train_test_split(ds.X, ds.y, test_size=test_size)

    # TODO - get more stats and pretty-print them. Maybe output to a debug file? 
    # TODO - progress bar for that sweet sweet seretonin
    def _cross_validate(self, estimator):
        scores = []
        kfold = KFold(shuffle=True).split(self.X_train, self.y_train)
        for k, (i_train, i_test) in enumerate(kfold):
            X_train, y_train = self.X_train.iloc[i_train], self.y_train.iloc[i_train]
            X_val, y_val = self.X_train.iloc[i_test], self.y_train.iloc[i_test]

            estimator.fit(X_train, y_train)
            yhat = estimator.predict(X_val)
            scores.append(estimator.score(X_val, y_val))

        print(scores)

