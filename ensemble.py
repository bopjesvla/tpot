from tpot.base import TPOTBase
from tpot import TPOTRegressor
from tpot.config.classifier import classifier_config_dict
from tpot.config.regressor import regressor_config_dict
from sklearn.datasets import load_boston
from sklearn import linear_model, model_selection
from sklearn.model_selection import train_test_split
import numpy as np
import multiprocessing as mult
import xgboost
from pmlb import fetch_data, regression_dataset_names

import time

i = 0
ests = []

def scorer(y_pred, y_true):
    return -np.mean((y_pred - y_true) ** 2)

def fit_predict(est, X_train, y_train, X_test):
    est.fit(X_train, y_train)
    return (est, est.predict(X_test))

def predict(est, X):
    return est.predict(X)

class EnsembleTPOTBase(TPOTBase):
    # def _check_periodic_pipeline(self):
    #     super()._check_periodic_pipeline()
    #     ensemble_pred = self.ensemble_predict(X_test)
    #     print("ensemble pred score:", scorer(ensemble_pred, y_test))
    #     # print(tpot.ensemble.coef_)
    def _evaluate_individuals(self, individuals, features, target, sample_weight=None, groups=None):
        operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts = self._preprocess_individuals(individuals)

        processes = None if self.n_jobs == -1 else self.n_jobs

        if type(self.cv) == int:
            self.cv = model_selection.KFold(self.cv)

        scores = []

        with mult.Pool(processes) as pool:
          for train, test in self.cv.split(features, target, groups):
              X_train, y_train, X_test, y_test = features[train], target[train], features[test], target[test]
              args = [(ind, X_train, y_train, X_test) for ind in sklearn_pipeline_list]
              estimators, predictions = zip(*pool.starmap(fit_predict, args))
              predictions = np.array(predictions).T
              ensembler = linear_model.Lasso(fit_intercept=False, positive=True, selection='random')
              ensembler.fit(predictions, y_test)
              self.ensemble = ensembler
              self.base_estimators = estimators
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
    ensemble_tpot = EnsembleTPOTRegressor(generations=10, population_size=10, verbosity=1, n_jobs=-1)
    normal_tpot = TPOTRegressor(generations=10, population_size=10, verbosity=1, n_jobs=-1)
    xgb = xgboost.XGBRegressor()

    try:
        ensemble_tpot.fit(X_train, y_train)
        print("fitted ensemble tpot")
        normal_tpot.fit(X_train, y_train)
        print("fitted normal tpot")
        xgb.fit(X_train, y_train)
        print("fitted xgboost")

        ensemble_pred = ensemble_tpot.ensemble_predict(X_test)
        ensemble_score = scorer(ensemble_pred, y_test)

        top_pred = ensemble_tpot.predict(X_test)
        top_score = scorer(top_pred, y_test)

        normal_pred = normal_tpot.predict(X_test)
        normal_score = scorer(normal_pred, y_test)

        xgb_pred = xgb.predict(X_test)
        xgb_score = scorer(xgb_pred, y_test)

        line = ','.join(str(x) for x in [regression_dataset, len(X), len(X[0]), ensemble_score, top_score, normal_score, xgb_score]) + '\n'
        print(line)
        x = open(f, 'a').write(line)
    except Exception as e:
        print(e)

#     # print(pred - y_train)
#     err = np.log(np.abs(pred - y_train) + .1)
#     err -= np.mean(err)
#     err /= np.std(err)
#     # w = np.maximum(1 + err, .5)
#     # print(w)

# tpot.export('tpot_iris_pipeline.py')
