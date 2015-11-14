
""" Generate performance plots for estimators in Application-Admission stage
game
"""

from __future__ import print_function, division
from os.path import join, dirname, realpath
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lawstructural.lawstructural.firststage as fs
import lawstructural.lawstructural.utils as lu
import lawstructural.lawstructural.bbl as bbl


#pylint: disable=too-few-public-methods
class Plotter(object):
    """ Plotter parent class for stages """
    def __init__(self, fname):
        self.fname = fname
        self.fig = plt.figure()
        self.plotnum = 1

    def out(self):
        """ Save plot """
        fpath = dirname(dirname(dirname(realpath(__file__))))
        fpath = join(fpath, 'Results', 'FirstStage', 'Figures',
                     self.fname + '.pdf')
        self.fig.savefig(fpath, format='pdf')


class StagePlotter(Plotter):
    """ Child class for stages """
    def __init__(self, fname):
        super(StagePlotter, self).__init__(fname)

    def estimator_plot(self, stage, key):
        """ Plotting function for any given stage """
        feature_importance = stage.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance /
                                      feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        axis = self.fig.add_subplot(2, 2, self.plotnum)
        axis.barh(pos, feature_importance[sorted_idx], align='center')
        axis.set_yticks(pos)
        rhs = np.array(lu.student_problem_vars())
        rhs[rhs == 'OverallRank'] = 'Rank'
        rhs[rhs == 'LSDAS_GPA'] = 'GPA'
        rhs[rhs == 'year'] = 'Year'
        axis.set_yticklabels(rhs[sorted_idx].tolist())
        axis.set_xlabel('Relative Importance: {0}'.format(key))
        self.plotnum += 1


class OutcomePlotter(Plotter):
    """ Child class for outcomes """
    def __init__(self, fname):
        super(OutcomePlotter, self).__init__(fname)

    def estimator_plot(self, stage, key):
        """ Plotting function for any given stage """
        feature_importance = stage.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance /
                                      feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        axis = self.fig.add_subplot(2, 2, self.plotnum)
        axis.barh(pos, feature_importance[sorted_idx], align='center')
        axis.set_yticks(pos)
        rhs = np.array(['Rank', 'Tuition'])
        axis.set_yticklabels(rhs[sorted_idx].tolist())
        axis.set_xlabel('Relative Importance: {0}'.format(key))
        self.plotnum += 1


class TuitionPlotter(Plotter):
    """ Child class for tuition """
    def __init__(self, fname):
        super(TuitionPlotter, self).__init__(fname)

    def estimator_plot(self, stage, key):
        """ Plotting function for any given stage """
        feature_importance = stage.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance /
                                      feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5
        axis = self.fig.add_subplot(1, 1, self.plotnum)
        axis.barh(pos, feature_importance[sorted_idx], align='center')
        axis.set_yticks(pos)
        rhs = np.array(['Rank', 'Treat', 'Year'])
        axis.set_yticklabels(rhs[sorted_idx].tolist())
        axis.set_xlabel('Relative Importance: {0}'.format(key))
        self.plotnum += 1


def plot_driver(est_dict, fname, plot_type):
    """ Drive usage of Plotter given a dictionary of estimates """
    plotter = plot_type(fname)
    for key, est in est_dict.iteritems():
        plotter.estimator_plot(est, key)
    plotter.out()



def main():
    """ Driver function """
    data_path = join(dirname(dirname(__file__)), 'data')
    data = pd.read_csv(join(data_path, 'lsnLong.csv'))
    data = data.dropna()
    data_p = bbl.BBL(None).data
    #pylint: disable=maybe-no-member
    data = data.loc[data.year <= 2013, :]
    firststage = fs.FirstStage(
        data_p, 'simple', {'verbose': 0, 'generate': 0}
    )
    stages = firststage.lsn_long_est(data)
    plot_driver(stages, 'appadmit_importance', StagePlotter)
    outcomes = firststage.application_admission()
    plot_driver(outcomes['treat0'], 'appadmit_outcomes_treat0_importance',
                OutcomePlotter)
    plot_driver(outcomes['treat1'], 'appadmit_outcomes_treat1_importance',
                OutcomePlotter)
    tuition = firststage.reaction()['Tuition']
    plot_driver({'Tuition': tuition}, 'fs_tuition_importance', TuitionPlotter)

