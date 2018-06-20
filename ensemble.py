from tpot.base import TPOTBase
from tpot import TPOTRegressor
from tpot.config.classifier import classifier_config_dict
from tpot.config.regressor import regressor_config_dict
from sklearn.datasets import load_boston
from sklearn import linear_model, model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import check_scoring
import numpy as np
import multiprocessing as mult
import xgboost
from pmlb import fetch_data, regression_dataset_names

import time

i = 0
ests = []

def scorer(est, X, y_true):
    return -np.mean((est.predict(X) - y_true) ** 2)

from stopit import ThreadingTimeout

def fit_predict_score(est, X_train, y_train, X_test, y_test, scoring_fn):
    with ThreadingTimeout(10) as ctx:
        est.fit(X_train, y_train)
        preds = est.predict(X_test)
        score = check_scoring(est, scoring=scoring_fn)(est, X, y)
        res = (est, preds, score)

    if ctx.state == ctx.EXECUTED:
        return res
    else:
        print('timed out')
        return (est, np.array([-10 for _ in X_test]), -float('inf'))

def predict(est, X):
    return est.predict(X)

class EnsembleTPOTBase(TPOTBase):
    def _check_periodic_pipeline(self):
        super()._check_periodic_pipeline()
        print("ensemble pred score:", scorer(EnsemblePredictor(self), X_test, y_test))
        # print(tpot.ensemble.coef_)
    def _evaluate_individuals(self, individuals, features, target, sample_weight=None, groups=None):
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = self._preprocess_individuals(individuals)

        processes = None if self.n_jobs == -1 else self.n_jobs

        if type(self.cv) == int:
            self.cv = model_selection.KFold(self.cv)

        scores = []
        ensembles = []
        coefs = []

        with mult.Pool(processes) as pool:
          for train, test in self.cv.split(features, target, groups):
              X_train, y_train, X_test, y_test = features[train], target[train], features[test], target[test]
              print(X_train.shape)
              args = [(ind, X_train, y_train, X_test, y_test, self.scoring_function) for ind in sklearn_pipeline_list]
              estimators, predictions, s = zip(*pool.starmap(fit_predict_score, args))
              # best_estimators_i = np.argpartition(s, -5)[-5:]
              # best_estimators = np.array(estimators)[best_estimators_i]
              predictions = np.array(predictions).T
              ensembler = linear_model.Lasso(alpha=0.01, fit_intercept=False, positive=True, selection='random')
              ensembler.fit(predictions, y_test)
              self.ensemble = ensembler
              scores.append(ensembler.coef_)

        import sys
        # sys.stdout.write("\nhey" + str(len(individuals)) + "\n")
        result_score_list = np.mean(np.array(scores), axis=0)
        ensembler.coef_ = result_score_list

        self._update_evaluated_individuals_(result_score_list, eval_individuals_str, operator_counts, stats_dicts)

        """Look up the operator count and cross validation score to use in the optimization"""
        return [(self.evaluated_individuals_[str(individual)]['operator_count'],
                self.evaluated_individuals_[str(individual)]['internal_cv_score'])
                for individual in individuals]
    def ensemble_predict(self, X):
        processes = None if self.n_jobs == -1 else self.n_jobs
        with mult.Pool(processes) as pool:
            predictions = pool.starmap(predict, [(est, X) for est in self.base_estimators])
            predictions = np.array(predictions).T
            return self.ensemble.predict(predictions)

class EnsemblePredictor():
    def __init__(self, tpot):
        self.tpot = tpot
    def predict(self, X):
        return self.tpot.ensemble_predict(X)

class EnsembleTPOTClassifier(EnsembleTPOTBase):
    """TPOT estimator for classification problems."""

    scoring_function = 'accuracy'  # Classification scoring
    default_config_dict = classifier_config_dict  # Classification dictionary
    classification = True
    regression = False


class EnsembleTPOTRegressor(EnsembleTPOTBase):
    """TPOT estimator for regression problems."""

    scoring_function = 'neg_mean_squared_error'  # Regression scoring
    default_config_dict = regressor_config_dict  # Regression dictionary
    classification = False
    regression = True

f = 'results' + str(time.time()) + '.csv'
is_mv = False

for i, regression_dataset in enumerate(regression_dataset_names):
    # if not is_mv:
    #     if regression_dataset != '344_mv':
    #         continue
    #     else:
    #         is_mv = True
    if '_fri_' in regression_dataset:
        continue

    X, y = fetch_data(regression_dataset, return_X_y=True)

    if len(X) > 100000:
        continue

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)
    print("name:", regression_dataset, "instances:", len(X), "features:", len(X[0]))
    ensemble_tpot = EnsembleTPOTRegressor(generations=100, population_size=100, verbosity=1, n_jobs=-1, scoring=scorer)
    normal_tpot = TPOTRegressor(generations=100, population_size=100, verbosity=1, n_jobs=-1, scoring=scorer)
    xgb = xgboost.XGBRegressor()

    try:
        ensemble_tpot.fit(X_train, y_train)
        print("fitted ensemble tpot")
        normal_tpot.fit(X_train, y_train)
        print("fitted normal tpot")
        xgb.fit(X_train, y_train)
        print("fitted xgboost")

        xgb_score = scorer(xgb, X_test, y_test)
        normal_score = scorer(normal_tpot, X_test, y_test)
        top_score = scorer(ensemble_tpot, X_test, y_test)
        ensemble_score = scorer(EnsemblePredictor(ensemble_tpot), X_test, y_test)

        line = ','.join(str(x) for x in [regression_dataset, len(X), len(X[0]), ensemble_score, top_score, normal_score, xgb_score]) + '\n'
        print(line)
        x = open(f, 'a').write(line)
    except Exception as e:
        raise e
        print(e)

#     # print(pred - y_train)
#     err = np.log(np.abs(pred - y_train) + .1)
#     err -= np.mean(err)
#     err /= np.std(err)
#     # w = np.maximum(1 + err, .5)
#     # print(w)

# tpot.export('tpot_iris_pipeline.py')
