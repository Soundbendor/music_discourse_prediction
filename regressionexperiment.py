from experiment import Experiment
from dataset import Dataset

from sklearn.model_selection import train_test_split

class RegressionExperiment(Experiment):
    
    def train_test_split(self, ds: Dataset, test_size: int):
        return train_test_split(ds.X, ds.y, test_size=test_size)