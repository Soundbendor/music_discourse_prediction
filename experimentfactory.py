import re
import csv

from configparser import ConfigParser
from pydoc import locate

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
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline


class ExperimentFactory:

    def __init__(self, config: str) -> None:
        cfp = ConfigParser()
        cfp.read(config)
        self.config = cfp
            
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

    def get_grid_search(self) -> dict:
        gs_args = self._get_arg('GS_PARAMS')
        param_grid = self.get_param_grid()

        return {
            'grid_search': gs_args['grid_search'] == 'true',
            'scoring': gs_args['scoring'],
            'param_grid': param_grid,
            'cv': int(gs_args['cv'])
        }


    def get_param_grid(self) -> list:
        pgrid_args = self._get_arg('PARAM_GRID')
        return [{k: self.ini_range(v)} for k, v in pgrid_args.items()]
        

    def ini_range(self, v):
        args = [x for x in v.split(',')]
        tp = locate(args.pop(0))
        return [tp(x) for x in args]


    def get_sampling_strategy(self):
        sampling_args = self._get_arg('SAMPLER')
        samplers = {
            'under_sample': RandomUnderSampler(sampling_strategy='majority'),
            'smote': KMeansSMOTE(sampling_strategy='not majority', cluster_balance_threshold=0.1, k_neighbors=5),
            'none': None
        }
        return samplers[sampling_args['sampling']]

    def get_model(self) -> BaseEstimator:
        model_config = self._get_arg('MODEL')
        
        supported_models = {
            'linearregressor': LinearRegression(),
            'randomforestregressor': RandomForestRegressor(n_jobs=-1),
            'kneighborsregressor': KNeighborsRegressor(), 
            'mlpregressor': MLPRegressor(verbose=True)
        }

        return supported_models[model_config['model']]

    def get_feature_selection_strategy(self):
        fselect_config = self._get_arg('FEATURE_SELECTION')        
        percent = fselect_config['percent_features']

        supported_selection_methods = {
            'pca': PCA(float(fselect_config['n_components'])),
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
            'stratify': self._stratify(),
            'valence_key': self.config['CONTROL']['valence_key'], 
            'arousal_key': self.config['CONTROL']['arousal_key'], 
            'meta_cols': re.sub(r"\s+", "", self.config['PREPROCESSING']['meta_cols']).split(',')
        }

    def get_y_keys(self) -> list:
        return [self.config['CONTROL']['valence_key'], self.config['CONTROL']['arousal_key']]

    def _stratify(self) -> bool:
        return self.config['CONTROL']['experiment_type'] == 'classification'
