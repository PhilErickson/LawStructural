
""" Contains primary class for estimation """

from __future__ import print_function, division
import os
import pandas as pd
import numpy as np
import lawstructural.lawstructural.firststage as fs
import lawstructural.lawstructural.secondstage as ss
import lawstructural.lawstructural.policysim as lps
import lawstructural.lawstructural.studentwelfare as sw


class BBL(object):
    """ Primary class for BBL-based estimation and simulation """

    def __init__(self, opts):
        self.opts = opts
        self.dir_name = os.path.dirname(os.path.dirname(__file__))
        self.tcount = 0  # Number of the current table being generated
        self.fs_params = {'react': {}, 'entrance': np.array([]),
                          'wage': np.array([]), 'rank': np.array([]),
                          'ranked': np.array([])}
        self.ss_params = np.array([])

        #self._dir_gen()
        data_path = os.path.join(self.dir_name, 'data', 'lawData.csv')
        self.data = pd.read_csv(data_path)
        #pylint: disable=maybe-no-member
        self.data = self.data.loc[self.data['year'] >= 2001, :]
        self.data = self.data.loc[self.data['OverallRank'] < 195, :]

    def data_check(self):
        """ Verify that data has been generated """
        path = os.path.join(self.dir_name, 'data', 'lawData.csv')
        if os.path.isfile(path):
            pass
        else:
            raise RuntimeError('Data not found. Rerun with reformatting.')

    def _firststage(self):
        """ Generate first stage estimates """
        print("ESTIMATING FIRST STAGE")
        first = fs.FirstStage(self.data, self.opts['reaction'], self.opts)
        self.fs_params['react'] = first.reaction()
        self.fs_params['entry'] = first.entrance()
        self.fs_params['rank'] = first.rank_evolve()
        appadmit = first.application_admission()
        self.fs_params['appadmit_student'] = appadmit['student']
        self.fs_params['appadmit'] = appadmit['period']
        self.fs_params['appadmit_stages'] = appadmit['stages']
        self.fs_params['student_sigmasq'] = 1
        print("FIRST STAGE ESTIMATED")

    def _secondstage(self):
        """ Generate second stage estimates """
        print("ESTIMATING SECOND STAGE")
        prob2 = ss.SecondStage(self.data, self.fs_params, self.opts)
        prob2.estimate()
        self.ss_params = prob2.ss_params
        #print("Now, run the .so function'
        #ls.simw(react=self.fs_params['react'], wage=self.fs_params['wage'], \
        #        rank=self.fs_params['rank'], ranked=self.fs_params['ranked'])
        #self.ss_params = ss.sim_q(coef)
        #ss.sim_q(self.fs_params, self.data)
        print("SECOND STAGE ESTIMATED")

    def estimate(self):
        """ Estimate model """
        print("* ---------------------- *")
        print("*    ESTIMATING MODEL    *")
        print("* ---------------------- *")
        self._firststage()
        if self.opts['generate']:
            self._secondstage()
        else:
            print("USING PREVIOUS SECOND STAGE ESTIMATES")
            try:
                dir_name = os.path.dirname(self.dir_name)
                data_path = os.path.join(dir_name, 'Results', 'SecondStage',
                                         'SSEst.npy')
                self.ss_params = np.load(data_path)
            except IOError:
                raise RuntimeError("Second stage results not yet generated. "
                                   "Rerun with option 'g'.")
        print("* ------------------------- *")
        print("*    ESTIMATION COMPLETE    *")
        print("* ------------------------- *")

    def simulate(self):
        """ Simulate counterfactuals """
        print("* -------------------------------- *")
        print("*    SIMULATING COUNTERFACTUALS    *")
        print("* -------------------------------- *")
        counter_sim = lps.CounterfactualSim(self.data, self.fs_params,
                                            self.ss_params, self.opts)
        diff_producer_surplus = counter_sim.simulate()
        student_welfare = sw.StudentWelfare(
            self.fs_params['react']['Tuition'],
            self.fs_params['student_sigmasq'],
            self.opts
        )
        diff_consumer_surplus = student_welfare.policy_comp()
        print("* ------------------------------- *")
        print("*    COUNTERFACTUALS SIMULATED    *")
        print("* ------------------------------- *")
        diff = diff_producer_surplus + diff_consumer_surplus
        print("TOTAL CHANGE IN WELFARE: {0}".format(diff))



