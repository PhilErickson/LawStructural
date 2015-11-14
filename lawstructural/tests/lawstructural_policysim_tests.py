#pylint: skip-file
""" Unit tests for policysim """

from __future__ import division, print_function
from nose.tools import *
import numpy as np
import pandas as pd
from os.path import dirname, join
from lawstructural.lawstructural.bbl import BBL
import lawstructural.lawstructural.policysim as lps
import lawstructural.lawstructural.constants as lc

class TestCounterfactualSim(object):
    """ Testing base class for Counterfactual Simulator """
    
    def __init__(self):
        opts = {'tables': 0, 'reaction': 'simple', 'entrance': 1, 'verbose': 1}
        bbl = BBL(opts)
        data = bbl.data
        bbl._firststage()
        self.fs_params = bbl.fs_params
        path = dirname(dirname(dirname(__file__)))
        path = join(path, 'Results', 'SecondStage', 'SSEst.npy')
        ss_params = np.load(path)
        self.countersim = lps.CounterfactualSim(data, self.fs_params,
                                                ss_params, opts)
        
    
    def resimulate(self, treat):
        sim_opts =  {'rank': self.countersim.rank_init,
                     'state': self.countersim.state_init,
                     'treat': treat}
        self.countersim.sim.simulate(
            sim_opts,
            perturb=self.countersim.perturb_gen(self.countersim.rank_init)
        )
        

class TestNormalizeRevenue(TestCounterfactualSim):
    """ Testing class for normalize_revenue """
    
    def __init__(self):
        super(TestNormalizeRevenue, self).__init__()
    
    def gen_results(self, treat):
        """ Generate results, both normalized and not """
        self.resimulate(treat)
        results = self.countersim.sim.get_results()
        revenue = results['revenue']
        revenue_normalize = self.countersim.normalize_revenue(results)
        beta = lc.BETA**np.arange(self.countersim.constants['n_periods'])
        return np.dot(revenue, beta), revenue_normalize
    
    def gen_diff(self, state):
        """ Generate expected difference in value for a state """
        results = self.countersim.sim.get_results()
        alpha = self.fs_params['demand'][state]
        beta = lc.BETA**np.arange(self.countersim.constants['n_periods'])
        state_revenue = np.dot(results['Tuition'], beta) * alpha
        state_revenue = state_revenue[
            np.where(self.countersim.state_init == state)
        ]
        return state_revenue
    
    def base_normalize_revenue_state(self, treat, state):
        """ Base method to test normalize_revenue for a given state """
        revenue, revenue_normalize = self.gen_results(treat)
        state_revenue = self.gen_diff(state)
        mask = np.array(self.countersim.state_init == state)
        diff = revenue[mask] - np.array(revenue_normalize)[mask]
        return state_revenue, diff
    
    def test_normalize_revenue_notreat_CA(self):
        """ Test for no treat/CA """
        state_revenue, diff = self.base_normalize_revenue_state(0, 'CA')
        assert np.allclose(state_revenue, diff)
    
    def test_normalize_revenue_notreat_UT(self):
        """ Test for no treat/UT """
        state_revenue, diff = self.base_normalize_revenue_state(0, 'UT')
        assert np.allclose(state_revenue, diff)

    def test_normalize_revenue_treat_CA(self):
        state_revenue, diff = self.base_normalize_revenue_state(1, 'CA')
        assert np.allclose(state_revenue, diff)

    def test_normalize_revenue_treat_UT(self):
        state_revenue, diff = self.base_normalize_revenue_state(1, 'UT')
        assert np.allclose(state_revenue, diff)
    
    def test_normalize_revenue_notreat_rank1(self):
        revenue, revenue_normalize = self.gen_results(0)
        mask = np.array(self.countersim.rank_init == 4)
        print("Rev: ", revenue[mask])
        print("Rev_norm: ", revenue_normalize[mask])
        #print("Ranks: ", self.countersim.rank_init)
        #unq, unq_idx = np.unique(self.countersim.rank_init,
        #                         return_inverse=True)
        #unq_cnt = np.bincount(unq_idx)
        #cnt_mask = unq_cnt > 1
        #dups = unq[cnt_mask]
        #print("Repeats: ", dups)
        #assert np.allclose(revenue_normalize[mask], revenue_normalize[mask][0])
        assert False

#def run():
#    t = TestNormalizeRevenue()
#    t.test_normalize_revenue_notreat_state()
#
#run()
