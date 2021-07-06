from experimentfactory import ExperimentFactory
from dataset import Dataset

from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



class Experiment:
    def __init__(self, dataset: Dataset, config: ExperimentFactory) -> None:
        self.ds = dataset
        self.config = config

    def run_experiment(self):
        sampler = self.config.get_sampling_strategy()
        model = self.config.get_model()
        fs = self.config.get_feature_selection_strategy()
        pipe = self._build_pipeline(fs, model, sampler)

    def _build_pipeline(self, feature_selection, sampling_method, model):
        return ImbPipeline([
            ('StandardScaler', StandardScaler()),
            ('FeatureSelection', feature_selection),
            ('Sampler', sampling_method),
            ('Model', model)
        ])

    def _build_grid_search(self, estimator):
        gs_args = self.config.get_grid_search()

        return GridSearchCV(
            estimator= estimator, 
            param_grid= self.config,
            refit= True,
            cv= gs_args['cv'],
            scoring= gs_args['scoring']
        )

# bulid an estimator
# build a grid search with that estimator (if relevant)
# run the grid search to return the best parameters for that param grid
# build a cross validator with the pipeline created
# run cross validation
# create final predictions
# output result graphics