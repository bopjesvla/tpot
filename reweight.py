from tpot import TPOTClassifier, TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

iris = load_boston()
X_train, X_test, y_train, y_test = train_test_split(iris.data.astype(np.float64),
    iris.target.astype(np.float64), train_size=0.75, test_size=0.25)

# global weights = None


# def weighted_scorer(scorer):
    # def wrapped_scorer(y_true, y_pred):

tpot = TPOTRegressor(generations=1, population_size=50, verbosity=2, n_jobs=-1, warm_start=True)
w = np.ones(X_train.shape[0])

for gen in range(5):
    tpot.fit(X_train, y_train, w)
    pred = tpot.predict(X_train)
    # print(pred - y_train)
    err = np.log(np.abs(pred - y_train) + .1)
    err -= np.mean(err)
    err /= np.std(err)
    # w = np.maximum(1 + err, .5)
    # print(w)

print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py')
