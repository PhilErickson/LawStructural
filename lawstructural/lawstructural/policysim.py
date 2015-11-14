""" Simulate counterfactual dynamic market effects of different policies """

from __future__ import print_function, division
import os
from os.path import join, dirname
import math
import numpy as np
import pandas as pd
from rpy2.robjects import r, pandas2ri, numpy2ri
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from copy import deepcopy
import lawstructural.lawstructural.constants as lc
import lawstructural.lawstructural.secondstage as ss
import lawstructural.lawstructural.utils as lu

r.library('splines')
pandas2ri.activate()
numpy2ri.activate()

class CounterfactualSim(object):
    """ Primary class for running different policy simulations """

    def __init__(self, data, fs_params, ss_params, opts):
        max_year = np.max(data.year)
        self.rank_init = data.OverallRank[
            data.year == max_year
        ].reset_index(drop=True)
        self.state_init = data.state[
            data.year == max_year
        ].reset_index(drop=True)
        self.constants = {'n_periods': 25, 'n_sims': lc.N_SIMS}
        self.sim = ss.Simulation(fs_params, self.constants, opts, ss_params)
        self.fs_params = fs_params
        self.opts = opts

    @staticmethod
    def quantile_list_gen():
        """ Generate list of quantiles and column names """
        quantiles = [5, 15, 25, 35, 50, 65, 75, 85, 95]
        columns = ['q' + str(el) for el in quantiles]
        return quantiles, columns

    def quantile_dynamics(self, sim_in):
        """ Calculate dynamics of quantiles for simulated variables

        Paramters
        ---------
        sim_in: dict; each member is a full market simulation for a variable
                with dimensions lc.N_SCHOOLS x self.constants['n_periods']
        """
        quantiles, columns = self.quantile_list_gen()
        dataframe = pd.DataFrame(
            np.zeros((self.constants['n_periods'], len(columns))),
            columns=columns
        )
        out = {}
        for key in sim_in:
            out[key] = deepcopy(dataframe)
            for i in xrange(len(quantiles)):
                out[key][columns[i]] = np.percentile(sim_in[key],
                                                     quantiles[i],
                                                     axis=0)
        return out

    @staticmethod
    def rank_list_gen():
        """ Generate rank intervals and column names """
        intervals = np.rint(np.linspace(1, lc.RBAR + 1, num=10))
        columns = [str(el) for el in intervals]
        for i in xrange(len(columns)):
            try:
                columns[i] = columns[i] + "-" + str(intervals[i + 1])
            except IndexError:
                columns[i] = columns[i] + "+"

        return intervals, columns

    def rank_dynamics(self, sim_in):
        """ Calculate dynamics of rank-conditional quantiles """
        intervals, columns = self.rank_list_gen()
        masks = np.zeros((len(self.rank_init), len(intervals)))
        for i in xrange(len(intervals)):
            try:
                masks[:, i] = ((self.rank_init >= intervals[i]) &
                               (self.rank_init < intervals[i + 1]))
            except IndexError:
                masks[:, i] = (self.rank_init >= intervals[i])
        masks = masks.astype(bool)
        dataframe = pd.DataFrame(
            np.zeros((self.constants['n_periods'], len(intervals))),
            columns=columns
        )
        out = {}
        for key in sim_in:
            out[key] = deepcopy(dataframe)
            for i in xrange(len(intervals)):
                val = sim_in[key][masks[:, i]]
                out[key][columns[i]] = np.mean(val, axis=0)

        return out

    def value_fun(self, results):
        """ Simulate value function data """
        beta = lc.BETA**np.arange(self.constants['n_periods'])
        val = np.dot(results['revenue'], beta)
        return pd.DataFrame({'OverallRank': np.array(self.rank_init),
                             'val': np.array(val)})

    def perturb_gen(self, rank):
        """ Generate perturbation inputs (zeros) for simulation

        Parameters
        ----------
        rank: ndarray
            Rank array in simulated market to be used for dimensions
        """
        pol_vars = lu.reaction_spec(self.opts['reaction'])[0]
        pol_vars.append('entry')
        perturb = np.zeros((len(rank), len(pol_vars)))
        return pd.DataFrame(perturb, columns=pol_vars)

    def basic_treat(self, treat=0):
        """ Basic market simulation under both treatment and none """
        self.sim.simulate(
            {'rank': self.rank_init, 'state': self.state_init, 'treat': treat},
            perturb=self.perturb_gen(self.rank_init),
            save_opts={'folder': os.path.join('Diagnostics', 'SimData')}
        )
        results = self.sim.get_results()
        #### START DIAGNOSTICS
        #for key in results:
        #    #print("Key: ", key)
        #    Diagnose(pd.DataFrame(results[key]),
        #             'basicsim_treat' + str(treat) + key).diagnose()
        #### END DIAGNOSTICS
        out = {}
        out['quantiles'] = self.quantile_dynamics(results)
        out['rank'] = self.rank_dynamics(results)
        out['value'] = self.value_fun(results)
        return out

    def simulate(self):
        """ Driver for running various market simulations """
        print("SIMULATING SCHOOL SIDE")
        print("    * Simulating without treatment")
        policy_a = self.basic_treat(0)
        print("    * Simulating with treatment")
        policy_b = self.basic_treat(1)
        print("GENERATING OUTPUT")
        comp = PolicyComp(policy_a, policy_b, "Dynamics")
        print("    * Quantile-conditional dynamics")
        comp.compare('quantiles')
        #print("    * Rank-conditional dynamics")
        #comp.compare('rank')
        print("    * Value functions")
        diff = comp.val()
        return diff


class PolicyComp(object):
    """ Runs comparisons on two sets of policies

    Parameters
    ----------
    policy_a: dict; output from given policy with members
        - quantiles: dict of DataFrame; dynamic evolution of raw quantiles
        - rank: dict of dict of DataFrames; dynamic evolution of rank-based
                quantiles
    policy_b: same as policy_a. Note that this one is the treatment policy
    """

    def __init__(self, policy_a, policy_b, folder):
        self.policy_a = policy_a
        self.policy_b = policy_b
        self.folder = folder
        self.out_dir = self.dir_gen()

    def dir_gen(self):
        """ Make sure results folders are generated """
        newdir =\
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(newdir, 'Results', 'SecondStage', 'Figures',
                            self.folder)
        if not os.path.exists(path):
            os.makedirs(path)
        return path


    def compare(self, comp_type):
        """ Compare rank-dependent means

        Parameters
        ----------
        comp_type: string; either "quantiles" or "rank"
        """
        for key in self.policy_a[comp_type]:
            columns = self.policy_a[comp_type][key].columns
            plt_dim = int(math.ceil(len(columns)**(.5)))
            fig, axes = plt.subplots(plt_dim, plt_dim,
                                     sharex=True, sharey=True)
            for i in xrange(plt_dim):
                for j in xrange(plt_dim):
                    plt_num = plt_dim * i + j
                    try:
                        axes[i, j].plot(
                            self.policy_a[comp_type][key][columns[plt_num]],
                            'k-', label='Base'
                        )
                        axes[i, j].plot(
                            self.policy_b[comp_type][key][columns[plt_num]],
                            'k,--', label='Treatment'
                        )
                        axes[i, j].set_title(columns[plt_num])
                    except IndexError:
                        continue

            no_info_line = mlines.Line2D([], [], color="black")
            info_line = mlines.Line2D([], [], linestyle='-.', color="black")
            #fig.legend('right', handles=[no_info_line, info_line])
            fig.legend((no_info_line, info_line), ("No Info", "Info"),
                       'lower right')
            fig.savefig(os.path.join(self.out_dir, key + comp_type + '.pdf'))

    @staticmethod
    def spline_est(data, new_data):
        """ Estimate conditional b-splines for value function """
        model = r.lm('val ~ bs(OverallRank, df=4)', data=data)
        return r.predict(model, newdata=new_data)


    def val(self):
        """ Estimate value functions with b-splines and compare """
        new_data = pd.DataFrame({'OverallRank': np.linspace(1, 194, 1000)})
        fit_a = self.spline_est(self.policy_a['value'], new_data)
        fit_b = self.spline_est(self.policy_b['value'], new_data)

        r.pdf(os.path.join(os.path.dirname(self.out_dir), 'value.pdf'))
        r.plot(new_data['OverallRank'], fit_a, type='l', xlab='Rank_M',
               ylab='V(Rank)')
        r.lines(new_data['OverallRank'], fit_b, col='red')
        r.points(self.policy_a['value']['OverallRank'],
                 self.policy_a['value']['val'],
                 col='black')
        r.points(self.policy_b['value']['OverallRank'],
                 self.policy_b['value']['val'],
                 col='red')
        r.legend('topright', np.array(['No Info', 'Info']),
                 lty=np.array([1, 1]), col=np.array(['black', 'red']))
        r('dev.off()')

        diff = np.array(fit_b) - np.array(fit_a)
        r.pdf(os.path.join(os.path.dirname(self.out_dir), 'value_diff.pdf'))
        r.plot(new_data['OverallRank'], diff, type='l', xlab='Rank',
               ylab='V(Rank|info=1) - V(Rank|info=0)')
        r.abline(h=0, lty=2)
        r('dev.off()')

        diff = (np.array(fit_b) - np.array(fit_a)) / np.array(fit_a)
        r.pdf(os.path.join(os.path.dirname(self.out_dir),
                           'value_percent_diff.pdf'))
        r.plot(new_data['OverallRank'], diff, type='l', xlab='Rank',
               ylab='(V(Rank|info=1) - V(Rank|info=0)) / V(Rank|info=0)')
        r.abline(h=0, lty=2)
        r('dev.off()')

        data_path = dirname(dirname(__file__))
        data_path = join(data_path, 'data', 'lawData.csv')
        data = pd.read_csv(data_path)
        new_data = deepcopy(data.loc[data['year'] == 2013, 'OverallRank'])
        #new_data = np.concatenate((
        #    new_data, np.zeros(lc.N_SCHOOLS - len(new_data))
        #))
        new_data = pd.DataFrame({'OverallRank': np.array(new_data)})
        fit_a = self.spline_est(self.policy_a['value'], new_data)
        fit_b = self.spline_est(self.policy_b['value'], new_data)
        diff = np.sum(np.array(fit_b) - np.array(fit_a))
        pdiff = diff / np.sum(fit_a)
        print("      - Change in Producer Surplus: {0}".format(diff))
        print("      - Percent change in Producer Surplus: {0}".format(pdiff))
        return diff



class Diagnose(object):
    """ Diagnose simulation results

    Parameters
    ----------
    data: pandas DataFrame
    axis: axis to do analysis across
    folder: destination folder for diagnostics output
    """

    def __init__(self, data, folder, axis=0):
        self.data = data
        self.folder = folder
        self.axis = axis
        self.out_dir = self.dir_gen()

    def dir_gen(self):
        """ Make sure diagnostics folders are generated """
        newdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        path = os.path.join(newdir, 'Results', 'Diagnostics', self.folder)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def get_figs(arr):
        """ Turn subplot array into graph array """
        def gfigs(fig):
            """ Figure extractor to be vectorized """
            return fig.get_figure()
        vec_gfigs = np.vectorize(gfigs, otypes=[object])
        return vec_gfigs(arr)

    @staticmethod
    def save_figs(arr, path):
        """ Save elements of graph array """
        def sfigs(fig, path):
            """ Saving function to be vectorized """
            fig.savefig(path)
        vec_sfigs = np.vectorize(sfigs, otypes=[object])
        vec_sfigs(arr, path)

    def diagnose(self):
        """ Run diagnostics """
        self.diag_hist()
        #self.diag_desc()

    def diag_hist(self):
        """ Make histogram of dataframe """
        fig = self.get_figs(self.data.hist())
        self.save_figs(fig, os.path.join(self.out_dir, 'hist.pdf'))

    def diag_desc(self):
        """ Print out descriptive statistics """
        print(self.data.describe())
