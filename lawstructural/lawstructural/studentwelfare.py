""" Module for computing student welfare changes based on information regime
change
"""

from __future__ import print_function, division
import numpy as np
from numpy.random import random_sample
import pandas as pd
from copy import deepcopy
from os.path import join, dirname
from sklearn import ensemble
from scipy.stats import norm  #pylint: disable=no-name-in-module
import lawstructural.lawstructural.utils as lu
import lawstructural.lawstructural.firststage as fs
import lawstructural.lawstructural.constants as lc

def gen_n_apps(treat):
    """ Generate average number of applications per year, conditional on
    information
    """
    if treat:
        years = range(2010, 2015)
    else:
        years = range(2000, 2010)
    n_apps = []
    for year in years:
        n_apps.append(fs.gen_n_apps(year).tolist()[0])
    return np.mean(n_apps)


class StudentWelfare(object):
    """ Primary class for the module

    Parameters
    ----------
    fs_params: dict
        estimates from firststage.lsn_long_est, to be used for admission and
        matriculation probabilities

    sigmasq: dict
        sigma-squared estimate for distribution of independent school-level
        values with keys 'treat0' and 'treat1' for estimates before and after
        information regime change
    """
    def __init__(self, fs_tuition, sigmasq, opts):
        self.fs_tuition = fs_tuition
        self.sigmasq = sigmasq
        self.opts = opts
        self.fs_rhs = lu.student_problem_vars()
        self.lsn_rhs = deepcopy(self.fs_rhs)
        self.lsn_rhs.remove('OverallRank')
        self.lsn_rhs.remove('Tuition')
        self.lsn_models = self.gen_matric_ev()
        self.data = None

    def gen_data(self, treat):
        """ Generate dataset of students for lc.N_PERIODS years. Variables
        include MedianLSAT, UndergraduatemedianGPA, year, treat
        """
        data_dir = join(dirname(dirname(__file__)), 'data')
        source_data = pd.read_csv(join(data_dir, 'lawschoolnumbers.csv'))
        source_data = source_data.loc[source_data.treat == treat]
        n_apps = gen_n_apps(treat)
        treat_years = {0: range(2003, 2010), 1: range(2010, 2014)}[treat]
        data = []
        for _ in xrange(2013, 2013 + lc.N_PERIODS):
            students = np.random.choice(
                source_data.loc[source_data.year.isin(treat_years), 'user'],
                size=n_apps
            )
            students = pd.DataFrame({
                'user': students[np.where(pd.notnull(students))]
            })
            data_year = pd.merge(
                students,
                source_data.loc[source_data.year.isin(treat_years)]
            )
            data.append(data_year)
        data = pd.concat(data)
        self.data = data.reset_index()

    def gen_matric_ev(self):
        """ Estimate expected rank of school j attended by student i given
        student i is in the final matriculant group
        """
        data_dir = join(dirname(dirname(__file__)), 'data')
        data = pd.read_csv(join(data_dir, 'lawschoolnumbers.csv'))
        models = {}
        models['admit_binary'] = ensemble.GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ).fit(data[self.lsn_rhs], data['admit_binary'])
        models['matric_binary'] = ensemble.GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ).fit(data.loc[data.admit_binary == 1, self.lsn_rhs],
              data.loc[data.admit_binary == 1, 'matric_binary'])
        params = {'n_estimators': 500, 'max_depth': 4,
                  'min_samples_split': 1, 'learning_rate': 0.01,
                  'loss': 'ls'}
        models['matric_ev'] = ensemble.GradientBoostingRegressor(**params)
        models['matric_ev'].fit(
            data.loc[data.matric_binary == 1, self.lsn_rhs],
            data.loc[data.matric_binary == 1, 'matric'])
        return models

    def predict_admit(self):
        """ Use first stage estimates to predict binary admission outcome """
        threshold = random_sample(self.data.shape[0])
        self.data['admit_hat'] = self.lsn_models['admit_binary'].predict_proba(
            self.data[self.lsn_rhs]
        )[:, 1]
        self.data['admit_hat'] = 1 * (self.data['admit_hat'] > threshold)

    def predict_matric(self):
        """ Use first stage estimates to predict matriculation outcomes
        """
        self.data.loc[self.data['admit_hat'] == 1, 'matric_hat_pr'] = \
            self.lsn_models['matric_binary'].predict_proba(
                self.data.loc[self.data['admit_hat'] == 1, self.lsn_rhs]
            )[:, 1]
        data_temp = deepcopy(self.data)  # use treatment probabilities for utl
        data_temp['treat'] = 1
        self.data.loc[self.data['admit_hat'] == 1, 'matric_hat_pr_treat'] = \
            self.lsn_models['matric_binary'].predict_proba(
                data_temp.loc[data_temp['admit_hat'] == 1, self.lsn_rhs]
            )[:, 1]


        self.data.loc[self.data['admit_hat'] == 1, 'OverallRank'] = \
            self.lsn_models['matric_ev'].predict(
                self.data.loc[self.data['admit_hat'] == 1, self.lsn_rhs]
            )
        self.data.loc[self.data['admit_hat'] == 1, 'Tuition'] = \
            self.fs_tuition.predict(
                self.data.loc[self.data['admit_hat'] == 1,
                              ['OverallRank', 'treat', 'year']]
            )


    def utility(self):
        """ Function derived from inverting probability of student i
        attending school j given school j is in admissions set
        """
        utility = norm.ppf(self.data['matric_hat_pr_treat'],
                           scale=self.sigmasq)
        utility = self.data['Tuition'] - utility

        return np.nansum(utility)

    def payoff_matric(self):
        """ Predict payoff from any given matriculation """
        pass

    def gen_welfare(self, treat):
        """ Generate welfare for the simulated population of students for
        given information regime.
        """
        self.gen_data(treat)
        self.predict_admit()
        self.predict_matric()
        payoff = self.utility()
        return payoff

    def policy_comp(self):
        """ Compare surplus with and without  """
        print("SIMULATING STUDENT SIDE")
        print("    * Simulating without treatment")
        payoff0 = self.gen_welfare(0)
        print("    * Simulating with treatment")
        payoff1 = self.gen_welfare(1)
        diff = payoff1 - payoff0
        pdiff = diff / payoff0
        print("      - Change in Consumer Surplus: {0}".format(diff))
        print("      - Percent change in Consumer Surplus: {0}".format(pdiff))
        return diff
