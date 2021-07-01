from configparser import ConfigParser

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.base import BaseEstimator

from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline as make_imb_pipe


class ExperimentFactory:

    def __init__(self, config: str) -> None:
        cfp = ConfigParser()
        cfp.read(config)
        self.config = cfp

    def build_experiment(self):
        self.control_args = self.get_control_args
        self.model = self.get_model()
        self.fs = self.get_feature_selection_strategy()
        self.sampler = self.get_sampling_strategy()
        self.ppargs = self.get_preprocessing_args()

        self.estimator = Pipeline(
            ('StandardScaler', StandardScaler()),
            ('FeatureSelection', self.fs),
            ('Sampler', self.sampler),
            ('Model', self.model)
        )
        self.grid_search = None

        if(self.control_args['grid_search'] == True):
            self.grid_search = self.get_grid_search()
            

    def _get_arg(self, key) -> dict:
        table = self.config[key]
        return {k: v.strip().lower() for k, v in table.items()}

    def get_control_args(self):
        control_args = self._get_arg('CONTROL')
        return {
            'grid_search': control_args['grid_search'] == 'true',
            'valence_key': self.config['CONTROL']['valence_key'],
            'arousal_key': self.config['CONTROL']['arousal_key'],
            'experiment_type': control_args['experiment_type']
        }

    def get_grid_search(self):
        pass

    def get_sampling_strategy(self):
        sampling_args = self.get_arg('SAMPLER')
        samplers = {
            'under_sample': RandomUnderSampler(sampling_strategy='majority'),
            'smote': make_imb_pipe(
                KMeansSMOTE(sampling_strategy='not majority', cluster_balance_threshold=0.1, k_neighbors=5),
            )
        }
        return samplers[sampling_args['sampling']]

    def get_model(self) -> BaseEstimator:
        model_config = self._get_arg('MODEL')
        
        supported_models = {
            'LinearRegressor': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=-1),
            'KNeighborsRegressor': KNeighborsRegressor(), 
            'MLPRegressor': MLPRegressor(verbose=True)
        }

        return supported_models[model_config['model']]

    def get_feature_selection_strategy(self):
        fselect_config = self._get_arg('FEATURE_SELECTION')        
        percent = fselect_config['percent_features']

        supported_selection_methods = {
            'PCA': PCA(fselect_config['n_components']),
            'f_regression': SelectPercentile(score_func=f_regression, percentile=percent), 
            'mutual_info_regression': SelectPercentile(score_func=mutual_info_regression, percentile=percent),
            'none': None
        }
        return supported_selection_methods[fselect_config['selector']]

    def get_preprocessing_args(self) -> dict:
        pre_config = self._get_arg('PREPROCESSING')
        return {
            'threshold': float(pre_config['threshold']), 
            'test_size': float(pre_config['test_size']), 
            'valence_key': self.config['CONTROL']['valence_key'], 
            'arousal_key': self.config['CONTROL']['arousal_key'], 
        }

