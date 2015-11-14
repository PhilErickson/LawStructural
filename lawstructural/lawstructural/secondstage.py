""" Second stage functions """

from __future__ import print_function, division
import os
import sys
import numpy as np
import pandas as pd
import multiprocessing as mp
from copy import copy, deepcopy
from rpy2.robjects import r, pandas2ri, numpy2ri
from scipy.stats import norm  #pylint: disable=no-name-in-module
import statsmodels.api as sm
#import lawstructural.ext.lawsimw as ls
import lawstructural.lawstructural.constants as lc
import lawstructural.lawstructural.utils as lu
import lawstructural.knitro.interface as ki

r.library('splines')
pandas2ri.activate()
numpy2ri.activate()

#TODO: Remove 'state' references

def rank_shock_std():
    """ Get std. dev. for the Tobit rank shock """
    dir_name = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(dir_name, 'data', 'lawData.csv')
    dataframe = pd.read_csv(data_path)
    #pylint: disable=maybe-no-member
    mask = ((dataframe.OverallRank < lc.UNRANKED) &
            (dataframe.OverallRankL < lc.UNRANKED))
    dfm = dataframe[mask]
    diff = dfm.OverallRank - dfm.OverallRankL
    return diff.std()


class Market(object):
    """ Class to keep track of period data and methods in the market

    Paramters
    ---------
    initial: dictionary of initial conditions with keys
        - rank: ndarray of ranks (zero for inactive firms)
        - state: ndarray of state locations for each school
        - treat: binary, 1 if information treatment in place
    firststage: dictionary; with keys
        - params: first-stage parameter estimates
        - perturb: pd.DataFrame, perturbation of policy functions
    shock: dictionary of random rank-relevant draws with keys
        - 'ev': drawn from N(0, sigma) for shock to rank for previously ranked,
                currently ranked schools
        - 'ev_tilde': same, but for previously unranked schools
    opts: dictionary of user options with keys
        - 'tables'
        - 'reaction'
        - 'entrance'
        - 'verbose'
    """

    def __init__(self, initial, firststage, shock, opts, secondstage=None):
        self.opts = opts
        self.data = pd.DataFrame(initial['rank'], columns=['OverallRank'])
        self.data['treat'] = initial['treat']
        self.data['RankTreat'] = initial['rank'] * initial['treat']
        self.data['GPATreat'] = initial['treat']
        self.active = np.array(self.data['OverallRank'] > 0)
        self.data['demand'] = np.zeros(initial['rank'].shape)
        self.firststage = deepcopy(firststage)
        # Select the functions associated with correct information structure
        self.firststage['params']['appadmit'] = \
            self.firststage['params']['appadmit'][
                'treat' + str(initial['treat'])
            ]
        self.secondstage = secondstage
        if np.any(self.secondstage):
            self.struc_param_unpacker()
        self.shock = shock
        self.reac_vars = lu.reaction_spec(opts['reaction'])[0]
        for rvar in self.reac_vars:
            self.data[rvar] = np.zeros(initial['rank'].shape)
            self.data[rvar + '_comp'] = np.zeros(initial['rank'].shape)

        self.data['revenue'] = self.data['demand'] * self.data['Tuition']
        self.add_donations()
        self.data['year'] = 2013
        self.data['const'] = 1

    def struc_param_unpacker(self):
        """ Unpacks structural parameters into dictionary """
        params = {}
        params['donations'] = self.secondstage
        self.secondstage = params

    def add_donations(self):
        """ Add net donations to revenue """
        if np.any(self.secondstage):
            self.data['revenue'][self.active] = (
                self.data['revenue'][self.active] +
                np.dot(
                    np.column_stack((
                        self.data['OverallRank'][self.active],
                        self.data['OverallRank'][self.active]**2
                    )),
                    self.secondstage['donations']
                )
            )

    def increment(self):
        """ Increment year """
        self.data['year'] += 1

    def reac_demand(self):
        """ Reaction variables to be used in demand function """
        reac_temp = copy(self.reac_vars)
        reac_temp.remove('Tuition')
        reac_temp.insert(0, 'tuition_hat')
        return reac_temp

    def enter(self):
        """ Simulate entrants and update self.active """
        # Update self.active
        pass

    def exit(self):
        """ Simulate exits """
        # Update self.active
        pass

    def reaction(self):
        """ Solve for optimal reaction-variable decisions of active schools """
        for rvar in ['Tuition']:
            self.data[rvar][self.active] = (
                self.firststage['params']['react'][rvar].predict(
                    self.data.loc[self.active, ['OverallRank', 'treat', 'year']]
                )
            )

            self.data[rvar][self.active] = (
                self.data[rvar][self.active] +
                self.firststage['perturb'][rvar][self.active]
            )

    def demand(self):
        """ Solve for demand given state and reactions """
        for lhs in ['demand', 'MedianLSAT', 'UndergraduatemedianGPA']:
            self.data[lhs][self.active] = \
                self.firststage['params']['appadmit'][lhs].predict(
                    self.data.loc[self.active,
                                  ['OverallRank', 'Tuition', 'year']]
                )
        self.data['demand'][self.active] = (
            self.data['demand'][self.active] * 5  # Adjust based on scaling
        )
        self.data['revenue'][self.active] = (self.data['demand'][self.active] *
                                             self.data['Tuition'][self.active])
        self.add_donations()

    def ev_tobit(self, tilde):
        """ Generate expected value for interior rank """
        period = self.data['year'][0] - 2013
        if tilde:
            beta = self.firststage['params']['rank']['beta_tilde_t']
            sigma = self.firststage['params']['rank']['sigma_tilde_t']
            shock = self.shock['ev_tilde'][self.active, period]
        else:
            beta = self.firststage['params']['rank']['beta_t']
            sigma = self.firststage['params']['rank']['sigma_t']
            shock = self.shock['ev'][self.active, period]
        x_vars = lu.rank_evol_vars(tilde=tilde, react=self.opts['reaction'],
                                   sim=True)
        xbeta = sm.add_constant(self.data[x_vars][self.active])
        xbeta = np.dot(xbeta, beta)
        lambda_tobit = ((norm.pdf((1 - xbeta + shock) / sigma) -
                         norm.pdf((lc.RBAR - xbeta + shock) / sigma)) /
                        (norm.cdf((lc.RBAR - xbeta + shock) / sigma) -
                         (norm.cdf((1 - xbeta + shock) / sigma))))
        out = xbeta + shock + sigma * lambda_tobit
        return out

    @staticmethod
    def ranksift(arr):
        """ Sift ranks into integers, skip ahead on ties """
        ind = 1
        base = np.unique(arr)  # Returned array is sorted
        arr_out = np.zeros(arr.shape)
        for element in base:
            arr_out[np.array(arr == element)] = ind
            ind = np.sum(arr == element) + ind
        return arr_out

    def rank_normalize(self):
        """ Normalize rank array to fill range. Starting at 1, ascend (descend)
        in number (quality)
        """
        #mask = np.array(self.active & (self.data['OverallRank'] <= lc.RBAR))
        rank_next = np.array(self.data['OverallRank'][self.active])
        rank_next = np.rint(rank_next)
        if np.any(np.where(rank_next < 1)[0]):
            rank_next[np.where(rank_next < 1)[0]] = 1
        self.data['OverallRank'][self.active] = self.ranksift(rank_next)

    def rank_evolution(self):
        """ Solve for rank evolution of the active schools"""
        rank_next = self.ev_tobit(tilde=0)
        self.data['OverallRank'][self.active] = rank_next
        self.rank_normalize()
        self.data['RankTreat'] = self.data['OverallRank'] * self.data['treat']

    def get_data(self):
        """ Return current values of data in the market """
        return self.data


class Simulation(object):
    """ Methods/attributes for keeping track of market history and performing
    simulation for given inputs

    Parameters
    ----------
    fs_params: dictionary; first-stage estimates
    ss_params: ndarray; second-stage estimates
        - Note: this will only be non-empty for policy evaluation,
                not estimation
    constant: dictionary; constants for simulations with keys
        - 'n_periods': number of periods to simulate value function
        - 'n_sims': number of value function simulations
    opts: dictionary of user options with keys
        - 'tables'
        - 'reaction'
        - 'entrance'
        - 'verbose'
    """

    def __init__(self, fs_params, constants, opts, ss_params=None):
        self.fs_params = fs_params
        self.ss_params = ss_params
        self.constants = constants
        self.opts = opts
        self.reac_vars = lu.reaction_spec(opts['reaction'])[0]
        self.out = {'demand': np.array([]),
                    'OverallRank': np.array([]),
                    'revenue': np.array([])}
        for rvar in self.reac_vars:
            self.out[rvar] = np.array([])

    def init_output(self, rank):
        """ Initialize output arrays to be correct size """
        for key in self.out:
            self.out[key] = np.zeros((len(rank), self.constants['n_periods']))

    def shock_gen(self, rank):
        """ Generate shocks for simulation. For ev and ev_tilde, sigma=0.1 is
        about 0.5*sigma for the best ranked schools. This gives a lower-bound
        on rank transition variability, which is desirable since persistence is
        a marked characteristic of ranks. In the future, this could possibly be
        modified to allow for heteroskedastic shocks.

        Parameters:
        -----------
        rank: numpy array
            only used for length of shocks to be used

        Returns:
        --------
        shock: Dictionary with keys
            - 'ev': N(0, 1), dim = length of state vector x n_periods
            - 'ev_tilde': N(0, 1), dim = length of state vector x n_periods
        """
        shock = {}
        # Change to 1
        sigma = 0.5
        shock['ev'] = np.random.normal(
            loc=0.0,
            #scale=self.fs_params['rank']['sigma_t'],
            scale=sigma,
            size=(len(rank), self.constants['n_periods']))
        shock['ev_tilde'] = np.random.normal(
            loc=0.0,
            #scale=self.fs_params['rank']['sigma_tilde_t'],
            scale=sigma,
            size=(len(rank), self.constants['n_periods']))

        return shock


    def simulate(self, initial, perturb=None, save_opts=None):
        """ Main simulation routine

        Parameters
        ----------
        initial: dictionary of initial simulation values with keys
            - rank: ndarray of ranks (zero for inactive firms)
            - state: ndarray of state locations for each school
            - treat: binary, 1 if information treatment in place
        perturb: pd.DataFrame of policy perturbations
        save_opts: dictionary; options for saving output
            - 'save': Binary; 1 for saving output
            - 'folder': subfolder of 'Results' to save output in
        """
        np.random.seed(333453)  # Common Random Numbers (CRN)
        self.init_output(initial['rank'])
        sim_out = deepcopy(self.out)

        for _ in xrange(self.constants['n_sims']):
            shock = self.shock_gen(initial['rank'])
            market = Market(
                initial={'rank': initial['rank'], 'state': initial['state'],
                         'treat': initial['treat']},
                firststage={'params': self.fs_params, 'perturb': perturb},
                secondstage=self.ss_params,
                shock=shock, opts=self.opts
            )
            for period in xrange(self.constants['n_periods']):
                if self.opts['entrance']:
                    market.enter()
                    market.exit()
                market.reaction()
                market.demand()
                market.rank_evolution()
                market.increment()
                data = market.get_data()
                for key in self.out:
                    sim_out[key][:, period] = data[key]
            for key in self.out:
                self.out[key] = self.out[key] + sim_out[key]
        for key in self.out:
            self.out[key] = self.out[key] / self.constants['n_sims']
        if save_opts:
            savedir =\
                    os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            path = os.path.join(savedir, 'Results', save_opts['folder'])
            if not os.path.exists(path):
                os.makedirs(path)
            for key in self.out:
                np.savetxt(os.path.join(path, key + '.csv'), self.out[key],
                           delimiter=',')

    def get_results(self):
        """ Return results """
        return self.out


def simulation_thread_top_function(arg):
    """ Top function for pickling simulation thread """
    second_stage, job_args, policy = arg
    second_stage.simulation_thread(job_args, policy,
                                   second_stage.perturb['data'][policy])


class SecondStage(object):
    """ Class for second stage simulation and estimation

    Params
    ------
    data: Full data set (only year=2013 used - simulate from last year)
    fs_params: parameter estimates from first stage
    opts: standard option dictionary
    """

    def __init__(self, data, fs_params, opts):
        self.data = data.loc[
            data['year'] == 2013, :
        ].reset_index(drop=True)
        self.opts = opts
        if opts['entrance']:
            self.add_potentials()
        self.fs_params = fs_params
        self.ss_params = np.array([])
        # Randomly select schools to be perturbed for each alternate policy
        self.perturb = {
            'school': np.random.choice(range(self.data.shape[0]), lc.N_ALTPOL)
        }
        self.perturb['data'] = self.perturb_policy(data)
        manager = mp.Manager()
        self.basis = manager.list()
        #self.basis = []

    def add_potentials(self):
        """ Add potential entrants from random vector of states.
        Modifies self.data to include schools with rank=0
        """
        pass

    def perturb_policy(self, data):
        """ Generate perturbations for policy functions. Randomly picks one
        school per alternate policy and generates multiplicative perturbations
        for that school.

        Returns
        -------
            list of length lc.N_ALTPOL, with each element a pandas DataFrame,
            dimension nxp with n = (# of schools) and p = (# of equations
            to perturb). Each row is filled with ones, except the row
            corresponding to the perturbed school

        Note that the first element of the list will just be an array of ones
        """
        pol_vars = lu.reaction_spec(self.opts['reaction'])[0]
        # Multiplicative perturbations to have same SE as raw data
        stdev = np.std(data[pol_vars])
        perturbs = np.zeros((self.data.shape[0], len(pol_vars)))
        perturbs = [pd.DataFrame(perturbs, columns=pol_vars)]
        for policy in xrange(1, lc.N_ALTPOL):
            # Use this line instead to have data-based perturbations
            #school = np.random.normal(1, stdev)
            school = np.random.normal(size=len(stdev))
            perturbs.append(deepcopy(perturbs[0]))
            perturbs[policy].loc[self.perturb['school'][policy], :] = school

        return perturbs

    def gen_basis(self, policy, optimal_data, alt_data):
        """ Generate basis functions. Append a basis function (list) to
        self.basis for the given alternate policy.

        Parameters
        ----------
        policy: int
            policy number (0 is the non-perturbed policy)
        optimal_data: dict
            Data generated from optimal policy with keys for each market
            variable and each member a numpy array
        alt_data: dict
            Same as optimal_data, but generated from alternate policy
        """
        school = self.perturb['school'][policy]
        beta = lc.BETA**np.arange(lc.N_PERIODS)
        basis = []

        # In the next two blocks, the ~np.isnan statements clean out artifacts
        # of simulation

        # Tuition Revenue
        diff = (alt_data['revenue'][school, :] -
                optimal_data['revenue'][school, :])
        basis.append(np.dot(diff[~np.isnan(diff)], beta[~np.isnan(diff)]))

        # Structural Component - Quadratic
        diff = (alt_data['OverallRank'][school, :] -
                optimal_data['OverallRank'][school, :])
        basis.append(np.dot(diff[~np.isnan(diff)], beta[~np.isnan(diff)]))
        diff = (alt_data['OverallRank'][school, :]**2 -
                optimal_data['OverallRank'][school, :]**2)
        basis.append(np.dot(diff[~np.isnan(diff)], beta[~np.isnan(diff)]))

        self.basis.append(np.array(basis))
        print(".", end="")
        sys.stdout.flush()

    def objective_q(self, theta):
        """ Minimum distance estimator objective function """
        func_g = np.dot(self.basis, np.hstack((1, theta)))
        out = (func_g > 0) * (func_g**2)
        return np.sum(out) / len(out)

    def min_q(self):
        """ Minimization routine for function 'Q', minimum distance estimator
        constructed from the inner product of the difference basis functions
        and the structural parameters, squared and indicated by
        non-equilibrium values.
        """
        guess = np.array([10000, -100])
        if self.opts['verbose']:
            ktr_opts = {}
        else:
            ktr_opts = {'outlev': 'none'}
        result = ki.ktrsolve(fun=self.objective_q, guess=guess,
                             options=ktr_opts, verbose=self.opts['verbose'])
        self.ss_params = result['coef']
        dir_name = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(dir_name, 'Results', 'SecondStage',
                                 'SSEst.npy')
        np.save(data_path, self.ss_params)

    def simulation_thread(self, thread_args, policy, perturb):
        """ All calls to be made in each thread for alternative policy
        simulation.
        """
        thread_args['sim'].simulate({'rank': thread_args['rank'],
                                     'state': thread_args['state'],
                                     'treat': thread_args['treat']},
                                    perturb)
        self.gen_basis(policy, thread_args['optimal_data'],
                       thread_args['sim'].out)

    def estimate(self):
        """ Primary driving function for estimation """
        constants = {'n_periods': lc.N_PERIODS, 'n_sims': lc.N_SIMS}
        treat = 0
        sim = Simulation(self.fs_params, constants, self.opts)
        print("    * Simulating alternate policies")
        #tic = os.times()
        rank = self.data.OverallRank
        #rank = np.concatenate((
        #    rank, np.zeros(lc.N_SCHOOLS - len(rank))
        #))
        state = self.data.state
        sim.simulate({'rank': rank, 'state': state, 'treat': treat},
                     self.perturb['data'][0])
        optimal_data = deepcopy(sim.out)
        # Set up for multiprocessing
        job_args = {'sim': sim, 'rank': rank, 'state': state,
                    'treat': treat, 'optimal_data': optimal_data}
        if self.opts['multiprocessing']:
            # Method 3: Parallelization with controlled thread count
            args = ((self, job_args, policy) for
                    policy in range(1, lc.N_ALTPOL))
            pool = mp.Pool(processes=lc.N_THREADS)
            pool.map(simulation_thread_top_function, args)
            pool.close()
            pool.join()
        else:
            # Method 1: No parallelization
            for policy in xrange(1, lc.N_ALTPOL):
                self.simulation_thread(job_args, policy,
                                       self.perturb['data'][policy])
        # Method 2: Naive parallelization
        #jobs = []
        #for policy in xrange(1, lc.N_ALTPOL):
        #    job = mp.Process(
        #        target=self.simulation_thread,
        #        args=(job_args, policy, self.perturb['data'][policy])
        #    )
        #    jobs.append(job)
        #    job.start()
        #for j in jobs:
        #    j.join()



        print()
        self.basis = [el for el in self.basis]
        self.basis = pd.DataFrame(self.basis,
                                  columns=['revenue', 'linear', 'quadratic'])
        self.min_q()

    def get_estimates(self):
        """ Returns structural parameter estimates """
        return self.ss_params

