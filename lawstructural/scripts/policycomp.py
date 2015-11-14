""" Compare methods for estimating policy functions """

from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn import ensemble, svm
from sklearn.metrics import mean_squared_error  #, r2_score
from rpy2.robjects import r, pandas2ri, numpy2ri

r.library('splines')
numpy2ri.activate()
pandas2ri.activate()


def get_xvars(rvar):
    """ Return variables to be used in estimation """
    xvars = ['OverallRank', 'treat']
    if rvar == 'Tuition':
        xvars.append('year')
    return xvars


def r_squared(d_true, d_pred):
    """ Return R^2 """
    ss_res = np.sum((d_true - d_pred)**2)
    ss_tot = np.sum((d_true - np.mean(d_true))**2)
    return 1 - (ss_res / ss_tot)


def analytics(rvar, data_type, d_true, d_pred):
    """ Generate MSE and R^2 for predictions

    Parameters
    ----------
    rvar: str
        estimation variable
    data_type: str
        type of data (Testing or Training)
    true: numpy array
        real data points
    pred: numpy array
        predicted values
    """
    print("    Variable: {0}".format(rvar))
    mse = mean_squared_error(d_true, d_pred)
    print("    %s MSE: %.4f" % (data_type, mse))
    #rsquared = r2_score(d_true, d_pred)
    rsquared = r_squared(d_true, d_pred)
    print("    %s R2: %.2f" % (data_type, rsquared))



def method_boosting(rvar, train, test):
    """ Calls to boosting method and performance statistics """
    print("BOOSTING")
    xvars = get_xvars(rvar)
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
              'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(train[xvars], train[rvar])
    analytics(rvar, 'Training', train[rvar], clf.predict(train[xvars]))
    if rvar != "UndergraduatemedianGPA":
        analytics(rvar, 'Testing', test[rvar], clf.predict(test[xvars]))
    print()


def method_svm(rvar, train, test):
    """ Support vector regression """
    print("SVM")
    xvars = get_xvars(rvar)
    clf = svm.SVR()
    clf.fit(train[xvars], train[rvar])
    analytics(rvar, 'Training', train[rvar], clf.predict(train[xvars]))
    if rvar != "UndergraduatemedianGPA":
        analytics(rvar, 'Testing', test[rvar], clf.predict(test[xvars]))
    print()


def method_spline(rvar, train, test):
    """ B-splines with interaction """
    print("Splines")
    formula = rvar + ' ~ bs(OverallRank, df=6) + treat + '\
              'treat:bs(OverallRank, df=6) - 1'
    if rvar == 'Tuition':
        formula = formula + ' + year'
    model = r.lm(formula, data=train)
    #print(r.summary(model).rx2('coefficients'))
    print(r.summary(model).rx2('r.squared'))
    #print(r.summary(model))
    analytics(rvar, 'Training', train[rvar],
              np.array(r.predict(model)))
    if rvar != "UndergraduatemedianGPA":
        analytics(rvar, 'Testing', test[rvar],
                  np.array(r.predict(model, newdata=test)))
    print()


def method_comp(rvar, data_in):
    """ Method to compare different methods
    Parameters
    ----------
    rvar: str
        reaction variable on which to test methods
    """
    data = deepcopy(data_in)
    data = data[[rvar, 'OverallRank', 'treat', 'year']]
    data = data.dropna()
    train = data.loc[data['year'] >= 2000, :]
    test = data.loc[data['year'] < 2000, :]
    method_boosting(rvar, train, test)
    method_svm(rvar, train, test)
    method_spline(rvar, train, test)


def main():
    """ Driver function """
    data_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_path = os.path.join(data_path, 'data', 'lawData.csv')
    data = pd.read_csv(data_path)
    for rvar in ['Tuition', 'UndergraduatemedianGPA', 'MedianLSAT']:
        print("* --------------------------------%s *" % ('-' * len(rvar)))
        print("*    COMPARISONS FOR VARIABLE: %s    *" % rvar)
        print("* --------------------------------%s *" % ('-' * len(rvar)))
        method_comp(rvar, data)


if __name__ == '__main__':
    main()
