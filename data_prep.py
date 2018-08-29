import pandas as pd
import numpy as np
import string
from sklearn.model_selection import train_test_split
import json

class DataPreparator:

    def __init__(self, conf_filepath):
        with open(conf_filepath) as conf_file:    
            self.conf = json.load(conf_file)

    def _load_text_data_frame(self):
        filepath = self.conf["FILEPATH"]
        self.text_data_frame = pd.read_csv(filepath, **self.conf["CSV_READER_ARGS"])

    def _transform(self, text_serie):
        '''
        lowercase and remove ponctuation of the 'text_serie' pandas serie

        '''
        text_serie = text_serie.map(lambda x: x.lower())
        
        

    def _get_text_features(self):
        self._transform(self.text_data_frame[self.conf["SENTENCE_COLUMN_NAME"]])
        text_features_data_frame = self.text_data_frame[self.conf["SENTENCE_COLUMN_NAME"]]
        return text_features_data_frame


    def _get_labels(self):
        return self.text_data_frame[self.conf["SENTIMENT_COLUMN_NAME"]]

    def _train_test_split(self):
        self._load_text_data_frame()
        X = self._get_text_features()
        y = self._get_labels()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        return X_train, X_test, y_train, y_test







    


