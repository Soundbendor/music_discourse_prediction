import re
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


class ExperimentFactory:

    def __init__(self, config: str) -> None:
        cfp = ConfigParser()
        cfp.read(config)
        self.config = cfp

    def get_model(self) -> BaseEstimator:
        model_config = self.config['MODEL']
        
        supported_models = {
            'LinearRegressor': LinearRegression(),
            'RandomForestRegressor': RandomForestRegressor(n_jobs=-1),
            'KNeighborsRegressor': KNeighborsRegressor(), 
            'MLPRegressor': MLPRegressor(verbose=True)
        }

        return supported_models[model_config['model']]

    def get_feature_selection_strategy(self):
        fselect_config = self.config['FEATURE_SELECTION']
        percent = fselect_config['percent_features']

        supported_selection_methods = {
            'PCA': PCA(fselect_config['n_components']),
            'f_regression': SelectPercentile(score_func=f_regression, percentile=percent), 
            'mutual_info_regression': SelectPercentile(score_func=mutual_info_regression, percentile=percent),
            'none': None
        }
        return supported_selection_methods[fselect_config['selector']]

    def get_preprocessing_args(self) -> dict:
        pre_config = self.config['PREPROCESSING']
        return {
            'threshold': float(pre_config['threshold']), 
            'test_size': float(pre_config['test_size']), 
            'valence_key': self.config['CONTROL']['valence_key'], 
            'arousal_key': self.config['CONTROL']['arousal_key'], 
            'meta_cols': re.sub(r"\s+", "", pre_config['meta_cols']).split(',')
        }

    def get_model_args(self) -> dict:
        return dict(self.config['MODEL_ARGS'])