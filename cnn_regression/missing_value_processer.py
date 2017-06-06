import numpy as np
import sys

# from fancyimpute import KNN
from sklearn.preprocessing import Imputer
# import theano

# import pickle
# import os

from config import root

root_path = root()


def missing_check(X_incomplete):
    X_incomplete = np.array(X_incomplete)
    if X_incomplete.ndim == 1:
        # X_incomplete = np.atleast_2d(X_incomplete)
        # X_incomplete = X_incomplete.T
        X_incomplete = X_incomplete.reshape(-1, 1)
    length = len(X_incomplete)

    for elem in range(length):
        if 'NaN' in X_incomplete[elem]:
            # missing_recover(X_incomplete[:, elem])
            X_incomplete = missing_recover(X_incomplete, elem)
    return np.array(X_incomplete, dtype=np.float)


def missing_recover(X, index, k=3):
    # X is the complete data matrix
    # X_incomplete has the same values as X except a subset have been replace with NaN

    # Use 3 nearest rows which have a feature to fill in each row's missing features
    # X_filled_knn = KNN(k=k).complete(X_incomplete)

    # return X_filled_knn

    # sklearn
    # index is the minimal row no. which has missing value in X.
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)

    forward = index - k if ((index - k) >= 0) else 0
    backward = index + k + 1 if ((index + k + 1) <= len(X)) else len(X)
    if index:
        imp.fit(X[forward:index].tolist()+X[index+1:backward].tolist())
    else:
        imp.fit(X[1:])

    try:
        X[index] = imp.transform(X[index].reshape(1, -1))
    except:
        print('Error: imputation, ', sys.exc_info()[0])
    return X
