#pylint: skip-file
""" Unit tests for secondstage """

import numpy as np
import pandas as pd
import lawstructural.lawstructural.firststage as fs
import lawstructural.lawstructural.utils as lu
from lawstructural.lawstructural.bbl import BBL


class TestReturns(object):
    """ Class for testing correct parameters are returned from firststage"""

    def __init__(self):
        self.opts = {'tables': 0, 'reaction': 'simple', 'entrance': 1,
                     'verbose': 1}
        self.bbl = BBL(self.opts['tables'], self.opts['reaction'],
                       self.opts['entrance'], self.opts['verbose'])
        self.bbl._firststage()
        self.fs_params = self.bbl.fs_params
    
    def test_demand(self):
        tvars = [
            'tuition_hat', 'MedianLSAT', 'UndergraduatemedianGPA',
            'OverallRank', 'treat', 'RankTreat', 'AL', 'AR', 'WY'
        ]
        expected = np.array([
            -0.927284, 19.286605, -203.475186, -1.167404, -22.917197,
            0.354199, -1749.431023, -1799.828207, -1875.056317
        ])
        print("Actual")
        print(self.fs_params['demand'][tvars])
        assert np.allclose(self.fs_params['demand'][tvars], expected)

    def test_react(self):
        pass

    def test_entry(self):
        pass

    def test_rank(self):
        pass
