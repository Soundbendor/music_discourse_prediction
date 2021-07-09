from experiment import Experiment
from dataset import Dataset

from sklearn.model_selection import train_test_split

'''
WARNING - This class is not ready yet!
The y-label in dataset needs to be changed to be classes, not regression values!
'''

class ClassificationExperiment(Experiment):
    
    def train_test_split(self, ds: Dataset, test_size: int):
        raise Exception("Need to fix y-labeling in dataset. Currently non-functional.")
        print("DEBUG - I'm stratifying the training split")
        return train_test_split(ds.X, ds.y, stratify=ds.y, test_size=test_size)

    # TODO - Cross-validation. Similar to reg-exp's cv, but use StratifiedKFold
