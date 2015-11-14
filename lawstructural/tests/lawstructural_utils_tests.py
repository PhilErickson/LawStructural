#pylint: skip-file
""" Unit tests for lawutils """

from nose.tools import *
import numpy as np
import pandas as pd
import lawstructural.lawstructural.utils as lu

class TestCompMatGen(object):
    def __init__(self):
        self.rank = np.arange(5)

    def test_1(self):
        expected = np.array([[0., 1., 0., 0., 0.],
                             [0.5, 0., 0.5, 0., 0.],
                             [0., 0.5, 0., 0.5, 0.],
                             [0., 0., 0.5, 0., 0.5],
                             [0., 0., 0., 1., 0.]])
        assert np.all(expected == lu.comp_mat_gen(self.rank, 1))

    def test_5(self):
        expected = np.array([
            [0., 0.25, 0.25, 0.25, 0.25],
            [0.25, 0., 0.25, 0.25, 0.25],
            [0.25, 0.25, 0., 0.25, 0.25],
            [0.25, 0.25, 0.25, 0., 0.25],
            [0.25, 0.25, 0.25, 0.25, 0.]])
        assert np.all(expected == lu.comp_mat_gen(self.rank, 5))



