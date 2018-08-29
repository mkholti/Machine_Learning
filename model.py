import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from data_prep import DataPreparator
import json

class Model:
    def __init__(self, conf_filepath):
        self.conf_filepath = conf_filepath
        with open(conf_filepath) as conf_file:    
            self.conf = json.load(conf_file)
        self.encoder = TfidfVectorizer(**self.conf["PARAM_VECTORIZER"])
        self.classifier = LogisticRegression(**self.conf["PARAM_CLASSIFIER"])

    def _make_pipeline(self):
        self.pipeline = Pipeline([('vectorizer', self.encoder),
                                  ('classifier', self.classifier)])

    def _grid_search(self):
        self._make_pipeline()
        dict_param_grid = self.conf["PARAM_GRID"]
        C = np.linspace(*dict_param_grid["LINSPACE_ARGS"])
        tfidf_min_df = dict_param_grid["tfidf_min_df"]
        tfidf_ngram_range = dict_param_grid["tfidf_ngram_range"]
        tfidf_max_df = dict_param_grid["tfidf_max_df"]
        tfidf_use_idf = dict_param_grid["tfidf_use_idf"]
        tfidf_sublinear_tf = dict_param_grid["tfidf_sublinear_tf"]
        param_grid = dict(classifier__C=C, vectorizer__min_df=tfidf_min_df, vectorizer__ngram_range=tfidf_ngram_range,
                            vectorizer__max_df=tfidf_max_df, vectorizer__use_idf=tfidf_use_idf, vectorizer__sublinear_tf=tfidf_sublinear_tf)
        grid = GridSearchCV(self.pipeline, param_grid=param_grid)
        return grid

    def _get_best_parameters(self):
        grid = self._grid_search()
        return grid.best_params_

    def _update_conf_file_parameters(self):
        best_parameters = self._get_best_parameters()
        self.conf['PARAM_CLASSIFIER']['C'] = best_parameters['classifier__C']
        self.conf['PARAM_VECTORIZER']['sublinear_tf'] = best_parameters['vectorizer__tfidf_sublinear_tf']
        self.conf['PARAM_VECTORIZER']['tfidf_min_df'] = best_parameters['vectorizer__tfidf_min_df']
        self.conf['PARAM_VECTORIZER']['tfidf_ngram_range'] = best_parameters['vectorizer__tfidf_ngram_range']
        self.conf['PARAM_VECTORIZER']['tfidf_max_df'] = best_parameters['vectorizer__tfidf_max_df']
        self.conf['PARAM_VECTORIZER']['tfidf_use_idf'] = best_parameters['vectorizer__tfidf_use_idf']


    def train(self):
        model = self._grid_search()
        X_train, y_train = DataPreparator(self.conf_filepath)._train_test_split()[0:4:2]
        print('fitting')
        model.fit(X_train, y_train)
    

    def score_report(self):
        model = self._grid_search()
        X_test, y_test = DataPreparator(self.conf_filepath)._train_test_split()[1:4:2]
        y_predict = model.predict(X_test)
        report = classification_report(y_test, y_predict)
        print(report)


