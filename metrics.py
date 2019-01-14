import numpy as np


def rmse(true, predicted):
    n = true.shape[0]
    return np.sqrt(
        1 / n * np.sum(np.power(true - predicted, 2))
    )


def mse(true, predicted):
    n = true.shape[0]
    return np.sqrt(
        1 / n * np.sum(np.abs(true - predicted))
    )


def dcg_score(r, k=5):
    r = np.asfarray(r)[:k]
    return np.sum(r / np.log2(np.arange(2, r.size + 2)))


def ndcg_score(r, k=5):
    dcg_max = dcg_score(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_score(r, k) / dcg_max


def gini(r):
    r = r.flatten()
    if np.amin(r) < 0:
        r -= np.amin(r)
    r += 0.0000001
    r = np.sort(r)
    indices = np.arange(1, r.shape[0] + 1)
    n = r.shape[0]
    return (np.sum((2 * indices - n - 1) * r)) / (n * np.sum(r))

