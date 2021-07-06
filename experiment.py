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

        pipe = self._build_pipeline(fs, sampler, model)
        best_est = self._run_grid_search(pipe)

    def _build_pipeline(self, feature_selection, sampling_method, model):
        return ImbPipeline([
            ('standardscaler', StandardScaler()),
            ('featureselection', feature_selection),
            ('sampler', sampling_method),
            ('model', model)
        ])

    def _run_grid_search(self, estimator):
        gs_args = self.config.get_grid_search()
        if(not gs_args['grid_search']):
            return estimator

        gs = self._build_grid_search(estimator, gs_args)
        gs = gs.fit(self.ds.X_train, self.ds.y_train)
        return gs.best_estimator_


    def _build_grid_search(self, estimator, gs_args: dict):
        print(estimator.get_params().keys())

        return GridSearchCV(
            estimator= estimator, 
            param_grid= gs_args['param_grid'],
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