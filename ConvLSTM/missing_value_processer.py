import numpy as np
import sys
from sklearn.preprocessing import Imputer
from config import root


root_path = root()


def missing_check(x_incomplete):

    if x_incomplete.ndim == 1:
        # X_incomplete = np.atleast_2d(X_incomplete)
        # X_incomplete = X_incomplete.T
        x_incomplete = x_incomplete.reshape(-1, 1)
    length = len(x_incomplete)

    for elem in range(length):
        if np.isnan(sum(sum(sum(x_incomplete[elem])))):
            x_incomplete = missing_recover(x_incomplete, elem)
    return np.array(x_incomplete, dtype=np.float)


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

    nan_indices = np.argwhere(np.isnan(X[index]))

    if X.ndim == 4:
        for nan_index in nan_indices:

            # Learn the relationship by using fit()
            if index > 0:
                imp.fit(np.vstack((X[forward:index, nan_index[0], nan_index[1]],
                                   X[index+1:backward, nan_index[0], nan_index[1]])))
            else:
                imp.fit(X[1:, nan_index[0], nan_index[1]])

            # Fill the missing value by using transform()
            try:
                X[index, nan_index[0], nan_index[1]] = \
                    imp.transform(X[index, nan_index[0], nan_index[1]].reshape(1, -1))
            except:
                print('Error: imputation, ', sys.exc_info()[0])
    else:

        # Learn the relationship by using fit()
        if index > 0:
            imp.fit(np.vstack((X[forward:index], X[index+1:backward])))
        else:
            imp.fit(X[1:])

        # Fill the missing value by using transform()
        try:
            X[index] = imp.transform(X[index].reshape(1, -1))
        except:
            print('Error: imputation, ', sys.exc_info()[0])

    tt = X[index, nan_index[0], nan_index[1]]

    return X
