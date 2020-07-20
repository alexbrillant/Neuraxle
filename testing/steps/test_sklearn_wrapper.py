import numpy as np
import pytest
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

from neuraxle.base import Identity
from neuraxle.steps.sklearn import SKLearnWrapper


def test_sklearn_wrapper_with_an_invalid_step():
    with pytest.raises(ValueError):
        SKLearnWrapper(Identity())


def test_sklearn_wrapper_fit_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_transform_with_predict():
    p = SKLearnWrapper(LinearRegression())
    data_inputs = np.expand_dims(np.array(list(range(10))), axis=-1)
    expected_outputs = np.expand_dims(np.array(list(range(10, 20))), axis=-1)

    p = p.fit(data_inputs, expected_outputs)
    outputs = p.transform(data_inputs)

    assert np.array_equal(outputs, expected_outputs)


def test_sklearn_wrapper_fit_transform_with_transform():
    n_components = 2
    p = SKLearnWrapper(PCA(n_components=n_components))
    dim1 = 10
    dim2 = 10
    data_inputs, expected_outputs = _create_data_source((dim1, dim2))

    p, outputs = p.fit_transform(data_inputs, expected_outputs)

    assert outputs.shape == (dim1, n_components)


def test_predict_proba():
    X, y = load_iris(return_X_y=True)
    clf = SKLearnWrapper(LogisticRegression(random_state=0).fit(X, y))
    proba = clf.predict_proba(X)

    assert proba.shape == (150,3)

def test_predict_log_proba():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)
    log_proba = clf.predict_log_proba(X[:2, :])

    assert log_proba.shape == (2,3)


def test_decision_function():
    X, y = load_iris(return_X_y=True)
    p = LogisticRegression(random_state=0).fit(X, y)
    decision_function = p.decision_function(X[:2, :])

    assert decision_function.shape == (2,3)


def test_partial_fit():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    p = SKLearnWrapper(SGDClassifier())

    p.partial_fit(X[0: int(len(X) / 10)], y[0: int(len(X) / 10)], classes=np.unique(y))
    first_score = p.score(X, y)
    p.partial_fit(X[int(len(X) / 10): -1], y[int(len(y) / 10): -1])
    second_score = p.score(X, y)

    assert second_score > first_score

def test_partial_fit_with_fit_args():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target
    p = SKLearnWrapper(SGDClassifier(), fit_args=dict(classes=np.unique(y)))

    p.partial_fit(X[0: int(len(X) / 10)], y[0: int(len(X) / 10)])
    first_score = p.score(X, y)
    p.partial_fit(X[int(len(X) / 10): -1], y[int(len(y) / 10): -1])
    second_score = p.score(X, y)

    assert second_score > first_score


def test_score():
    X, y = load_iris(return_X_y=True)
    clf = LogisticRegression(random_state=0).fit(X, y)

    score = clf.score(X, y)

    assert score > 0.10


def test_sparsify():
    X, y = load_iris(return_X_y=True)
    clf = SKLearnWrapper(LogisticRegression(random_state=0)).fit(X, y)

    coeff_sparse = clf.sparsify()

    assert isinstance(coeff_sparse, SKLearnWrapper)


def test_densify():
    X, y = load_iris(return_X_y=True)
    clf = SKLearnWrapper(LogisticRegression(random_state=0)).fit(X, y)

    coeff_dense = clf.densify()

    assert isinstance(coeff_dense, SKLearnWrapper)



def _create_data_source(shape):
    data_inputs = np.random.random(shape).astype(np.float32)
    expected_outputs = np.random.random(shape).astype(np.float32)
    return data_inputs, expected_outputs
