#pylint: skip-file
""" Unit tests for secondstage """

import numpy as np
import pandas as pd
import lawstructural.lawstructural.secondstage as ss
import lawstructural.lawstructural.utils as lu
from lawstructural.lawstructural.bbl import BBL


class Second(object):
    """ Base class for testing Second Stage classes/methods """

    def __init__(self):
        self.opts = {'tables': 0, 'reaction': 'simple', 'entrance': 1,
                     'verbose': 1}
        self.bbl = BBL(self.opts)
        self.data = self.bbl.data
        self.fs_params = self.first_stage()
        self.sim = ss.Simulation(
            fs_params=self.fs_params, ss_params=[],
            constants={'n_periods': 2, 'n_sims': 2},
            opts=self.opts
        )

    def first_stage(self):
        """ Get first-stage coefficients """
        self.bbl._firststage()
        return self.bbl.fs_params


class TestMarket(Second):
    """ Testing class for the Market class """

    def __init__(self):
        super(TestMarket, self).__init__()
        self.market = None
        np.random.seed(12332123453)
        self.reset_market()

    @staticmethod
    def shocks():
        shock = {
            'threshold': np.array([
                [ 0.19463642,  0.5873684 ,  0.78717283,  0.90054078],
                [ 0.94483933,  0.07396143,  0.98333879,  0.68228324],
                [ 0.42634376,  0.14129206,  0.84464217,  0.79256268],
                [ 0.20854527,  0.58906316,  0.05635544,  0.61617025],
                [ 0.50087842,  0.19310857,  0.08625367,  0.81300337],
                [ 0.41966183,  0.17935988,  0.52908728,  0.22612131],
                [ 0.12907962,  0.76501439,  0.84632002,  0.47479678]
            ]),
            'ev': np.array([
                [ 0.91115608,  0.25230844],
                [ 1.48955056, -0.2726139 ],
                [-0.52889126,  1.31421538],
                [-0.8749468 , -0.02523144],
                [ 0.62028673,  0.51080029],
                [ 0.01618796,  0.22108686],
                [ 1.38288772,  1.42322263]
            ]),
            'ev_tilde': np.array([
                [-1.18486114, -0.22821053],
                [-0.17726515, -0.16574475],
                [-0.02270774,  0.80759556],
                [ 0.31090053,  1.01443173],
                [ 0.96851244,  1.03848031],
                [-0.92384879, -0.00580604],
                [-0.72009631, -0.39224704]
            ])
        }
        return shock
    
    def perturb_gen(self, rank):
        """ Generate perturbation inputs """
        pol_vars = lu.reaction_spec(self.opts['reaction'])[0]
        pol_vars.append('entry')
        perturb = np.ones((len(rank), len(pol_vars)))
        return pd.DataFrame(perturb, columns=pol_vars)        
    
    def reset_market_fake(self, treat=0):
        """ Reset the market with fake data

        Parameters
        ----------
        treat: binary, indicates treatment period
        shock: binary, indicates alternative policy shock
        """
        initial = {
            'rank': np.array([1, 3, 14, 14, 18, 194, 195, 0]),
            'state': np.array(['AL', 'NY', 'IA', 'UT', 'TX', 'CA', 'MO', 'NC'])
        }
        np.random.seed(12332123453)
        perturb = self.perturb_gen(initial['rank'])
        shock = self.sim.shock_gen(initial['rank'])
        #shock = self.shocks()
        self.market = ss.Market(
            {'rank': initial['rank'], 'state': initial['state'],
             'treat': treat},
            {'params': self.fs_params, 'perturb': perturb}, shock, self.opts
        )

    def reset_market(self, treat=0):
        """ Reset the market with real data """
        data = self.data.loc[self.data.year == 2013, :].reset_index(drop=True)
        shock = self.sim.shock_gen(data['OverallRank'])
        perturb = self.perturb_gen(self.data['OverallRank'])
        self.market = ss.Market(
            {'rank': data['OverallRank'], 'state': data['state'],
             'treat': treat},
            {'params': self.fs_params, 'perturb': perturb}, shock, self.opts
        )
    
    def evolve_market(self, final_step):
        """ Evolve market until final_step """
        self.reset_market()
        steps = ['enter', 'exit', 'reaction', 'demand', 'rank_evolution']
        for step in steps:
            getattr(self.market, step)()
            if step == final_step:
                break

    def test_enter(self):
        pass

    def test_exit(self):
        pass

    def test_reaction(self):
        self.evolve_market('reaction')
        nrmse = []
        for rvar in lu.reaction_spec(self.opts['reaction'])[0]:
            mask = np.array(1 - np.isnan(
                self.data.loc[self.data['year'] == 2013, rvar]
            )).astype(bool)
            data = np.array(
                self.data.loc[self.data['year'] == 2013, rvar][mask]
            )
            rmse = (np.array(self.market.data[rvar][mask]) - data)**2
            rmse = np.sqrt(np.mean(rmse))
            # Normalize for percentage
            nrmse.append(rmse / (max(data) - min(data)))
        print("RMSE RVARS")
        print(nrmse)
        print("Expected")
        expected = np.array([0.12452261429473817, 
                             0.18392353736966649,
                             0.2677874577849565])
        assert np.allclose(np.array(nrmse), expected)

    def test_demand(self):
        self.evolve_market('demand')
        assert True

    def test_rank_normalize(self):
        self.reset_market_fake()
        self.market.rank_normalize()
        out = self.market.data['OverallRank']
        print(self.market.data['OverallRank'])
        assert np.all(out == np.array([1, 2, 3, 3, 5, 6, 7, 0]))

    def test_ranksift(self):
        self.reset_market_fake()
        arr = self.market.ranksift(np.array(self.market.data['OverallRank']))
        print("Arr")
        print(arr)
        assert np.all(arr == np.array([2, 3, 4, 4, 6, 7, 8, 1]))

    def test_rank_evolution(self):
        self.evolve_market('rank_evolution')
        data = np.array(
            self.data.loc[self.data['year'] == 2013, 'OverallRank']
        )
        rmse = (np.array(self.market.data['OverallRank']) - data)**2
        rmse = np.sqrt(np.mean(rmse))
        # Normalize for percentage
        rmse = rmse / (max(data) - min(data))
        print("RMSE Rank")
        print(rmse)
        print("Expected RMSE")
        expected = 0.0132467989693
        print(expected)
        assert np.allclose(rmse, expected)


#class TestSimulation(object):
#    """ Test the simulation routine """
#
#    def __init__(self):
#        self.opts = {'tables': 0, 'reaction': 'simple', 'entrance': 1,
#                     'verbose': 1}
#        self.bbl = BBL(self.opts['tables'], self.opts['reaction'],
#                       self.opts['entrance'], self.opts['verbose'])
#        self.data = self.bbl.data
#        self.coef = self.first_stage()
#
#    def first_stage(self):
#        """ Get first-stage coefficients """
#        self.bbl._firststage()
#        return self.bbl.fs_params
#
#    #def test_basic(self):
#    #    """ Test to make sure it works at all """
#    #    sim = ss.Simulation(2, self.coef, self.opts)
#    #    rank = np.array([1, 2, 4, 0])
#    #    state = ['UT', 'AL', 'IA', 'CA']
#    #    sim.simulate({'rank': rank, 'state': state, 'treat': 1})
#    #    out = sim.get_results()
#    #    assert True
#
#    #def test_large(self):
#    #    """ Test with large simulated rank/state vectors """
#    #    constants = {'n_periods': 50, 'n_sims': 5}
#    #    perturb = {}
#    #    for key in self.coef:
#    #        perturb[key] = np.array([0])
#    #    sim = ss.Simulation(self.coef, np.array([]), constants, self.opts)
#    #    n_draws = 200
#    #    np.random.seed(1876)
#    #    selec = np.random.choice(np.arange(self.data.shape[0]), n_draws)
#    #    rank = self.data.OverallRank[selec].reset_index(drop=True)
#    #    state = self.data.state[selec].reset_index(drop=True)
#    #    sim.simulate(rank, state, 1, perturb, {'save': 1, 'folder': 'test'})
#    #    out = sim.get_results()
#    #    assert True
