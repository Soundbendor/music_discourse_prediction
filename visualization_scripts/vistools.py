'''
An aggregator for visualization methods to be used programatically
Advise using this when calling visualization methods inside the prediction code
instead of importing other modules directly.

Includes vis models which can't/shouldn't be ran standalone. 
~Aidan B.
'''

import pandas as pd
from circumplex import _circumplex_model

class VisualizationTools:
    
    # def circumplex_model(data: pd.DataFrame, title, fname, val_key='Valence', aro_key='Arousal') -> None:
    #     _circumplex_model(data, title, fname, val_key, aro_key)