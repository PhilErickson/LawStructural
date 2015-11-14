""" First stage functions """

from __future__ import print_function, division
import os
import time
import pickle
from os.path import join, dirname
import pandas as pd
import numpy as np
from rpy2.robjects import r, pandas2ri, numpy2ri
import pandas.rpy.common as com
import multiprocessing as mp
from copy import copy, deepcopy
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
import lawstructural.lawstructural.utils as lu
import lawstructural.lawstructural.constants as lc

r.library('VGAM')
pandas2ri.activate()
numpy2ri.activate()


#TODO: add in tests for first-stage
class FirstStage(object):
    """ Estimate first stage functions for BBL """
    def __init__(self, data_p, react, opts):
        self.data_p = data_p
        self.react = react
        self.opts = opts

    def reaction(self):
        """ Estimate reaction functions """
        print("    * Estimating reaction functions")
        reac_vars = lu.reaction_spec(self.react)[0]
        models = {}
        data = copy(self.data_p[self.data_p['OverallRank'] <
                                np.max(self.data_p['OverallRank'])])
        for rvar in reac_vars:
            data_reac = data[['OverallRank', 'treat', 'year', rvar]]
            data_reac = data_reac.dropna()
            params = {'n_estimators': 500, 'max_depth': 4,
                      'min_samples_split': 1, 'learning_rate': 0.01,
                      'loss': 'ls'}
            model = ensemble.GradientBoostingRegressor(**params)
            model.fit(data_reac[['OverallRank', 'treat', 'year']],
                      data_reac[rvar])
            models[rvar] = model
            if self.opts['verbose']:
                print("--------- %s Policy Function Estimate ---------" % rvar)
                print("MSE (training): ", mean_squared_error(
                    data_reac[rvar],
                    model.predict(data_reac[['OverallRank', 'treat', 'year']])
                ))
        return models

    @staticmethod
    def entrance():
        """ Estimate entrance probability """
        print('    * Estimating entrance probability')
        data_path = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(data_path, 'data', 'entData.csv')
        data = pd.read_csv(data_path)
        model = r.glm('entry ~ treat', family='poisson', data=data)
        coef = model.rx2('coefficients')
        out = {0: np.exp(coef[0]), 1: np.exp(coef[0] + coef[1])}
        return out

    def rank_evolve(self):
        """ Estimate rank evolution through class RankEvolve """
        rank = RankEvolve(self.data_p, self.react, self.opts)
        return rank.solve()

    @staticmethod
    def _lsn_long_diagnostics(stages, data, rhs):
        """ Print out _lsn_long_est diagnostic report """
        print("---- Student problem measures of fit ----")
        for key in ['app', 'admit', 'matric']:
            if key == 'app':
                score_x = data[rhs]
                score_y = data['app']
            elif key == 'admit':
                score_x = data.loc[data['app'] == 1, rhs]
                score_y = data.loc[data['app'] == 1, 'admit']
            else:
                score_x = data.loc[data['admit'] == 1, rhs]
                score_y = data.loc[data['admit'] == 1, 'matric']
            score = stages[key].score(score_x, score_y)
            print("Score for {0} stage: {1}".format(key, score))

    def lsn_long_est(self, data):
        """ Application/admission estimates from lsnLong.csv """
        rhs = lu.student_problem_vars()
        stages = {}
        stages['app'] = ensemble.GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ).fit(data[rhs], data['app'])
        stages['admit'] = ensemble.GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ).fit(data.loc[data['app'] == 1, rhs],
              data.loc[data['app'] == 1, 'admit'])
        stages['matric'] = ensemble.GradientBoostingClassifier(
            n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0
        ).fit(data.loc[data['admit'] == 1, rhs],
              data.loc[data['admit'] == 1, 'matric'])
        if self.opts['verbose']:
            self._lsn_long_diagnostics(stages, data, rhs)
        return stages

    @staticmethod
    def parse_sims(sims):
        """ Turn results of simulate markets into bivariate splines """
        out = {}
        for key, data in sims.iteritems():
            out_key = {}
            for lhs in ['MedianLSAT', 'UndergraduatemedianGPA', 'demand']:
                params = {'n_estimators': 500, 'max_depth': 4,
                          'min_samples_split': 1, 'learning_rate': 0.01,
                          'loss': 'ls'}
                out_key[lhs] = ensemble.GradientBoostingRegressor(**params)
                out_key[lhs].fit(data[['OverallRank', 'Tuition', 'year']],
                                 data[lhs])
            out[key] = out_key
        return out

    def application_admission(self):
        """ Estimate application/admission subgame by Gradient Tree Boosting
        """
        print('    * Estimating application/admission parameters')
        data_path = join(dirname(dirname(__file__)), 'data')
        data = pd.read_csv(join(data_path, 'lsnLong.csv'))
        data = data.dropna()
        #pylint: disable=maybe-no-member
        data = data.loc[data.year <= 2013, :]
        stages = self.lsn_long_est(data)
        result_dir = join(dirname(dirname(dirname(__file__))), 'Results',
                          'FirstStage')
        if self.opts['generate']:
            appadmit = ApplicationAdmission(data, stages, self.opts)
            sims = appadmit.simulate()
            pickle.dump(sims, open(join(result_dir, 'sims.p'), 'wb'))
        else:
            try:
                sims = pd.read_pickle(join(result_dir, 'sims.p'))
            except IOError:
                raise RuntimeError("Stage game not yet simulated. "
                                   "Rerun with option 'g'.")
        for sim in sims:
            sims[sim].rename(columns={
                'LSAT': 'MedianLSAT',
                'LSDAS_GPA': 'UndergraduatemedianGPA',
            }, inplace=True)
        period_functions = self.parse_sims(sims)
        if self.opts['verbose']:
            print("Sim functions:")
            print(sims)
        return {'student': sims, 'period': period_functions, 'stages': stages}


def gen_data_task(args):
    """ Top-level function for generating each student to be used for
    multiprocessing
    """
    (app_data_year, students, keep_vars, out, i) = args
    out_data = app_data_year.loc[app_data_year['user'] == students[i],
                                 keep_vars]
    out_data['id'] = i
    out.append(out_data)
    return out


def gen_n_apps(year):
    """ Generate size of applicant pool """
    dpath = join(dirname(dirname(__file__)), 'data')
    data = pd.read_csv(join(dpath, 'abaeoy.csv'))
    return data.n_apps[data.year == year]


class ApplicationAdmission(object):
    """ Sift students into matriculations according to the
    Application/Admissions game estimate by the student's problem

    Parameters
    ----------
    data: pandas dataframe
        has columns 'OverallRank', 'Tuition', 'LSAT', 'LSDAS_GPA', 'treat',
        'year'
    firststage: dict
        first stage estimates
    opts: dict
        standard option dictionary
    """
    def __init__(self, app_data, firststage, opts):
        self.app_data = app_data
        self.data = None
        self.firststage = firststage
        self.opts = opts
        self.rhs = lu.student_problem_vars()

    def gen_data(self, treat):
        """ Generate random dataset """
        if treat:
            years = [2010, 2011, 2012]
        else:
            years = [2007, 2008, 2009]
        app_data_treat = self.app_data.loc[
            self.app_data['year'].isin(years)
        ].reset_index(drop=True)
        data = []
        for year in np.unique(app_data_treat['year']):
            app_data_year = self.app_data.loc[
                self.app_data['year'] == year
            ].reset_index(drop=True)
            students = np.random.choice(
                app_data_year['user'],
                size=(gen_n_apps(year).tolist()[0] / 5)  # Scale for memory
            )
            keep_vars = lu.student_problem_vars()
            keep_vars.append('school')
            #out = []
            out = mp.Manager().list()
            if self.opts['multiprocessing']:
                mp_args = ((app_data_year, students, keep_vars, out, i)
                           for i in range(len(students)))
                pool = mp.Pool(processes=lc.N_THREADS)
                pool.map(gen_data_task, mp_args)
                pool.close()
                pool.join()
            else:
                for i in xrange(len(students)):
                    out_data = app_data_year.loc[
                        app_data_year['user'] == students[i], keep_vars
                    ]
                    out_data['id'] = i
                    out.append(out_data)
            data_year = pd.concat(list(out))
            data_year.reset_index(inplace=True)
            data.append(data_year)
        data = pd.concat(data)
        data.reset_index(inplace=True)
        return data

    def _diag(self, lvar):
        """ Save diagnostic results """
        self.data = self.data.set_index('id')
        diag = self.data[lvar].sum(axis=0, level='id')
        diag.to_csv('AppAdmitDiag' + lvar + '.csv')
        self.data.reset_index(inplace=True)

    def application(self):
        """ Application stage """
        model = self.firststage['app']
        self.data['app'] = model.predict_proba(self.data[self.rhs])[:, 1]
        self.data['app'] = 1 * (
            self.data['app'] >
            np.random.uniform(size=len(self.data['app']))
        )
        # Uncomment to save output for diagnostics
        #self._diag('app')

    def admission(self):
        """ Admissions stage """
        model = self.firststage['admit']
        self.data.loc[self.data['app'] == 1, 'admit'] = model.predict_proba(
            self.data.loc[self.data['app'] == 1, self.rhs]
        )[:, 1]
        self.data.loc[self.data['app'] == 1, 'admit'] = 1 * (
            self.data.loc[self.data['app'] == 1, 'admit'] >
            np.random.uniform(size=np.sum(self.data['app']))
        )
        # Uncomment to save output for diagnostics
        #self._diag('admit')

    def matriculation(self):
        """ Matriculation stage """
        model = self.firststage['matric']
        self.data.loc[self.data['admit'] == 1, 'matric'] = model.predict_proba(
            self.data.loc[self.data['admit'] == 1, self.rhs]
        )[:, 1]
        #TODO: This would be the place to add matric shock
        self.data.loc[self.data['admit'] == 1, 'matric'] = self.data.loc[
            self.data['admit'] == 1, :
        ].groupby('id').apply(lambda x: x * (x == np.max(x)))['matric']
        self.data.loc[self.data['matric'] > 0, 'matric'] = 1 * (
            self.data.loc[self.data['matric'] > 0, 'matric'] >
            np.random.uniform(
                size=len(self.data.loc[self.data['matric'] > 0, 'matric'])
            )
        )
        # Uncomment to save output for diagnostics
        #self._diag('matric')

    def sim_run(self):
        """ One simulation run for getting LSAT, GPA, and demand """
        self.application()
        self.admission()
        self.matriculation()
        self.data = self.data.set_index(['school', 'year'])
        data_median = self.data.loc[
            self.data.matric == 1, :
        ].median(axis=0, level=['school', 'year'])
        data_sum = self.data.sum(axis=0, level=['school', 'year'])
        data_out = data_median[['OverallRank', 'Tuition', 'LSAT',
                                'LSDAS_GPA']]
        data_out['demand'] = data_sum['matric']
        data_out.reset_index(inplace=True)
        return data_out

    def simulate(self):
        """ Simulate LSAT, GPA, and demand functions from structural eqns """
        print("    * Simulating LSAT/GPA/Demand functions")
        out = {}
        time0 = time.time()
        for treat in [0, 1]:
            print("        - for treat={0}".format(treat))
            data_list = []
            data_base = self.gen_data(treat)
            for _ in xrange(lc.N_AESIMS):
                self.data = deepcopy(data_base)
                data_list.append(self.sim_run())
            data_list = pd.concat(data_list)
            out['treat' + str(treat)] = data_list.groupby(level=0).mean()
        if self.opts['verbose']:
            print("Simulation Run Time: {0}".format((time.time() - time0)))
        return out


class RankEvolve(object):
    """ Estimate rank evolution - tobit model with normalization in
    paper appendix.
    """
    def __init__(self, data_p, react, opts):
        self.react = react
        self.data = self.data_clean(data_p)
        self.opts = opts

    def data_clean(self, data):
        """ Clean NaNs out of necessary data """
        first = lu.rank_evol_vars(react=self.react)
        second = ['TopRanked', 'BottomRanked', 'InsideRanked', 'OverallRank',
                  'Ranked', 'RankedL']
        vars_keep = [first, second]
        vars_keep = [item for sublist in vars_keep for item in sublist]
        return data[vars_keep].dropna()

    def _r_tobit(self, data, xvars, rbar):
        """ Estimate tobit with function from r """
        r.assign('data', com.convert_to_r_dataframe(data))
        rhs = '+'.join(xvars)
        model = r("vglm(OverallRank ~ "+ rhs +", \
                          family=tobit(Upper=" + str(rbar) + ", Lower=1), \
                          data=data, crit='coeff')")
        if self.opts['verbose']:
            print(r.summary(model))
        out = r.coef(model, matrix=True)
        out = np.array(out)
        index = deepcopy(xvars)
        index.insert(0, 'const')
        beta = pd.Series(out[:, 0], index=index)
        return {'beta': beta, 'sigma': out[0, 1]}

    def problem(self):
        """ Problem definition for separated specification """
        x_vars = lu.rank_evol_vars(tilde=0, react=self.react)
        rbar = np.max(self.data.loc[self.data['Ranked'] > 0, 'OverallRank'])
        out = {}

        ## Data subsets
        data_r = self.data[self.data['RankedL'] > 0]
        data_rr = data_r[data_r['Ranked'] == 1]
        model_t = self._r_tobit(data_rr, x_vars, rbar)
        out['beta_t'] = model_t['beta']
        out['sigma_t'] = model_t['sigma']

        return out

    def solve(self):
        """ Solve rank evolution MLE """
        print('    * Estimating rank transition')
        out = self.problem()
        return out
