#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
Application of Ridge regression 
=========================================================
This example also shows the usefulness of applying Ridge regression to highly ill-conditioned matrices. 
For such matrices, a slight change in the target variable can cause huge variances in the calculated weights. 
In such cases, it is useful to set a certain regularization (alpha) to reduce this variation (noise).

"""

print(__doc__)

# Author : HangJie

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# X is the 10x10 Hilbert matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
y = np.ones(10)

###############################################################################
# Compute paths

n_alphas = 200
alphas = np.logspace(-10, -2, n_alphas)
clf = linear_model.Ridge(fit_intercept=False)

coefs = []
for a in alphas:
    clf.set_params(alpha=a)
    clf.fit(X, y)
    coefs.append(clf.coef_)

###############################################################################
# Display results

ax = plt.gca()

ax.plot(alphas, coefs)
ax.set_xscale('log')
ax.set_xlim(ax.get_xlim()[::-1])  # reverse axis
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()
