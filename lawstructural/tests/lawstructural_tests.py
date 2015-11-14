#pylint: skip-file
""" Unit tests for lawstructural """

#from nose.tools import *
#import numpy as np
#import pandas as pd
#import numpy.random as rand
#import statsmodels.api as sm
#from scipy.stats import norm
#from copy import copy
#import lawstructural.knitro.knitrosolver as ks
#import lawstructural.knitro.knitro as ktr
#from lawstructural.knitro.knitroNumPy import *
#import lawstructural.knitro.interface as ki
#import lawstructural.lawstructural.firststage as fs


#class TestTobit(object):
#    """ Testing KNITRO interface for tobit spec """
#    def __init__(self):
#        self.problem = {}
#        self.ktr_defs = {}
#        self.options = {}
#        self.sparse = {}
#        self.data = pd.DataFrame([])
#
#    def data_gen(self):
#        """ Generate basic tobit data """
#        np.random.seed(1876)
#        mu, sigma, n_sample = 0, 0.1, 1000
#        x_data = rand.uniform(1, 60, n_sample)
#        y_data = -18 + 2*x_data + rand.normal(mu, sigma, n_sample)
#        y_data[y_data < 1] = 1
#        y_data[y_data > 100] = 100
#        data = pd.DataFrame(np.array([y_data, x_data]).T,
#                            columns=['OverallRank', 'x'])
#        data['cons'] = 1
#        data['TopRanked'] = 1 * (data['OverallRank'] == 1)
#        data['BottomRanked'] = 1 * (data['OverallRank'] == 100)
#        data['InsideRanked'] = 1 * ((data['OverallRank'] > 1) &
#                                    (data['OverallRank'] < 100))
#        self.data = data
#
#    def tobit_dens(self, xbeta, sigma):
#        """ Standard logged tobit probability density function
#        Args:
#            - xbeta: array of (x_i'*beta)
#            - sigma: standard deviation
#        Returns:
#            density function evaluated at mu=xbeta
#        """
#        first = np.array(self.data['TopRanked']) * \
#                np.log(1 - norm.cdf(xbeta / sigma))
#        #print("First:", first)
#        second = np.array(self.data['BottomRanked']) * \
#                 norm.logcdf(xbeta / sigma)
#        #print("Second", second)
#        third = np.array(self.data['InsideRanked']) * \
#                (norm.logpdf((xbeta - self.data['OverallRank']) / sigma) -
#                 np.log(sigma))
#        #print("Third:", third)
#        return first + second + third
#
#    def xbeta_param_unpack(self, theta):
#        """ Unpacker for parameters and generator for xbeta """
#        beta = theta[:2]
#        sigma = theta[2]
#        x_vars = ['cons', 'x']
#        xbeta = np.dot(self.data[x_vars], beta)
#        return beta, sigma, xbeta
#
#    def set_problem(self):
#        """ Define objective """
#        def tobit_logl(theta):
#            """ Log likelihood for tobit density """
#            beta, sigma, xbeta = self.xbeta_param_unpack(theta)
#            tobit = self.tobit_dens(xbeta, sigma)
#            return -np.sum(tobit)
#
#        def constr(theta):
#            """ Constraint vector binding objective away from NaN evals """
#            beta, sigma, xbeta = self.xbeta_param_unpack(theta)
#            alpha = norm.ppf(1 - 1e-15)
#            return alpha*sigma - xbeta
#
#        self.problem['objective'] = tobit_logl
#        self.problem['constr'] = constr
#
#    def set_ktr_defs(self):
#        """ Set problem definitions from KNITRO Numpy example """
#        m_const = self.data.shape[0]
#        self.ktr_defs = {'bnds_lo': np.array([-ktr.KTR_INFBOUND,
#                                              -ktr.KTR_INFBOUND,
#                                              0]),
#                         'c_type': np.array([ktr.KTR_CONTYPE_LINEAR],
#                                            np.int64).repeat(m_const),
#                         'c_bnds_lo': np.zeros(m_const)}
#
#    def set_options(self):
#        """ Set problem options """
#        self.options = {'outlev': 'none', 'debug': 1}
#
#    def tobit_guess(self):
#        """ Generate initial guess for tobit model """
#        x_vars = ['cons', 'x']
#        guess_beta = sm.OLS(self.data['OverallRank'],
#                            self.data[x_vars]).fit()
#        guess_beta = guess_beta.params
#        guess_sigma = np.var(self.data['OverallRank']) * 10
#        return np.hstack((guess_beta, guess_sigma))
#
#    def test_tobit_mle(self):
#        """ Test tobit density with ktrinterface.interface """
#        self.data_gen()
#        self.set_problem()
#        guess = self.tobit_guess()
#        self.set_ktr_defs()
#        self.set_options()
#        result = ki.ktrsolve(fun=self.problem['objective'], guess=guess,
#                             constr=self.problem['constr'],
#                             ktr_defs=self.ktr_defs,
#                             options=self.options)
#        assert np.allclose(result['coef'],
#                           np.array([-17.552594, 1.92007581, 12.27937907]))
#
