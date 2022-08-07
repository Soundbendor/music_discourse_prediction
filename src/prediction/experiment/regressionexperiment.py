import pandas as pd

from .experiment import Experiment
from prediction.visualization import visualizations
from prediction.preprocessing.experimentset import ExperimentSet
from prediction.preprocessing.experimentfactory import ExperimentFactory
from prediction.preprocessing.dataset import Dataset

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error


class RegressionExperiment(Experiment):

    def __init__(self, dataset: Dataset, config: ExperimentFactory, output) -> None:
        super().__init__(dataset, config, output)
        self.metrics.append(mean_absolute_error)

    def split_dataset(self, ds: Dataset, test_size: int, key: str):
        X, y = ds.get_data(key)
        return train_test_split(X, y, test_size=test_size, random_state=128)

    def _get_k_fold(self, n_splits: int, expset: ExperimentSet):
        print(expset.X_train.shape)
        return KFold(n_splits, shuffle=True).split(expset.X_train, expset.y_train)

    def _get_keys(self) -> list:
        return [self.ds.val_key, self.ds.aro_key]

    def _generate_vis(self, df_pred: pd.DataFrame, df_results: pd.DataFrame) -> None:
        title = "Circumplex model of test subset"
        fname = "tmp/circumplex"
        visualizations.circumplex_model(
            df_pred, f"{title} - Predicted", f"{fname}_pred", self.ds.val_key, self.ds.aro_key)
        visualizations.circumplex_model(
            df_results, f"{title} - Actual", f"{fname}_actual", self.ds.val_key, self.ds.aro_key)
        self.report.set_circumplex(fname)
