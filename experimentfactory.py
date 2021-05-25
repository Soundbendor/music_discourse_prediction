from configparser import ConfigParser
import configparser

from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator

class ExperimentFactory:

    def __init__(self, config: ConfigParser) -> None:
        self.config = config

    def get_model(self) -> BaseEstimator:
        model_config = self.config['MODEL']
        
        supported_models = {
            'LinearRegressor': LinearRegression()
        }

        return supported_models[model_config['model']]