#pylint: disable=anomalous-backslash-in-string
""" Utility functions """
import re
import numpy as np
from os.path import join, dirname, abspath
from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage as STAP
#from rpy2.robjects import r
#from rpy2.robjects.vectors import ListVector
import lawstructural.lawstructural.constants as lc


def path_grab(folder, model):
    """ Return directory location for table results """
    par_name = dirname(dirname(dirname(__file__)))
    path = join(par_name, 'Results', folder, model)
    path = abspath(path)
    # Fix path for passing to R if windows machine
    if path[0] != '/':
        path = re.sub('\\\\', '\\\\\\\\', path)
    return path


def comp_mat_gen(rank, n_comp=10):
    """ Generate competition matrix for given rank vector """
    rank_1 = np.tile(rank, (rank.shape[0], 1)).T
    rank_2 = np.tile(rank, (rank.shape[0], 1))
    # Compare rank for every observation to every other obs
    rank_diff = rank_1 - rank_2
    # Indicate obs with rank +- n_comp of each obs
    rank_comp = ((rank_diff >= -n_comp) & (rank_diff <= n_comp)) * 1.0
    # Remove self
    rank_comp = rank_comp - np.eye(rank.shape[0])
    # Generate n's for averaging
    denom_comp = np.sum(rank_comp, 1)
    denom_comp = np.tile(denom_comp, (rank.shape[0], 1)).T
    # Turn elements into averaging weights
    rank_comp = rank_comp / denom_comp
    return rank_comp


def reaction_spec(react):
    """ Return reaction function variables and rank weightings"""
    if react == 'full':
        reac_vars = ['Tuition', 'p25LSATScore',
                     'MedianLSAT', 'p75LSATScore',
                     'p25UndergraduateGPA',
                     'UndergraduatemedianGPA',
                     'p75UndergraduateGPA']
        m_names = ['Tuition', '25\%LSAT', '50\%LSAT', '75\%LSAT', '25\%GPA',
                   '50\%GPA', '75\%GPA']
    elif react == 'simple':
        reac_vars = ['Tuition', 'MedianLSAT', 'UndergraduatemedianGPA']
        m_names = ['Tuition', '50\%LSAT', '50\%GPA']
        #reac_vars = ['Tuition']
        #m_names = ['Tuition']
    else:
        raise RuntimeError('Invalid reaction specification.')
    return [reac_vars, m_names]


def reaction_function(rvar):
    """ Return functional form for policy functions """
    function = '{0} ~ bs(OverallRank, df=' + str(lc.N_KNOTS) + ') + ' \
               'treat + treat:bs(OverallRank, df=' + str(lc.N_KNOTS) + ') - 1'
    return function.format(rvar)


#def demand_vars(reaction, states_raw=None):
#    """ Return RHS for demand estimation """
#    rvars = reaction_spec(reaction)[0]
#    rvars_comp = [rvar + '_comp' for rvar in rvars]
#    rvars.append('GPATreat')
#    #tuition = ['Tuition', 'Tuition_comp']
#    #states = np.delete(states_raw, np.where((states_raw == 'AK') |
#    #                                        (states_raw == 'NV')))
#    #states.sort()
#    rank = ['OverallRank', 'treat', 'RankTreat']
#    #dvars = [tuition, rank, ['year'], states]
#    dvars = [rvars, rank]
#    return [el for sublist in dvars for el in sublist]


def student_problem_vars():
    """ RHS variables for student problem estimation """
    return ['OverallRank', 'Tuition', 'LSAT', 'LSDAS_GPA', 'treat', 'year']


def fs_spline_vars():
    """ Variables to use as x, y in bivariate spline in stage game estimation
    """
    return {'x': 'OverallRank', 'y': 'Tuition'}


class StudentProblemVars(object):
    """ Return list of variables to be used in likelihood contribution for part
    lik
    """
    def __init__(self, lik):
        self.lik = lik

    @staticmethod
    def app():
        """ Application likelihood variables """
        return [
            'const', 'app_mean_tuition', 'year', 'app_mean_tuition_year',
            'LSAT', 'LSDAS_GPA', 'LSAT_treat', 'LSDAS_GPA_treat',
            'LSAT_LSDAS_GPA_treat'
        ]

    @staticmethod
    def admit():
        """ Admissions likelihood variables """
        return [
            'const', 'LSAT', 'LSDAS_GPA', 'app_mean_rank', 'app_std_rank',
            'app_n'
        ]

    @staticmethod
    def matric():
        """ Matriculation likelihood variables """
        return [
            'const', 'LSAT', 'LSDAS_GPA', 'admit_best', 'treat'
        ]

    def get(self):
        """ Return list of likelihood variables based on input """
        if self.lik == 'app':
            return self.app()
        elif self.lik == 'admit':
            return self.admit()
        elif self.lik == 'matric':
            return self.matric()
        else:
            raise RuntimeError('Invalid likelihood contributor')


def matric_vars(react):
    """ Return RHS for matriculation function """
    rvars = reaction_spec(react)[0]
    rvars.remove('Tuition')
    rvars_comp = [rvar + '_comp' for rvar in rvars]
    rank = ['OverallRank']
    mvars = [rvars, rvars_comp, rank]
    return [el for sublist in mvars for el in sublist]


#def demand_matric_vars(states, react):
#    """ Return all variables needed for demand/matriculation estimation
#    Returns
#    -------
#    dictionary; with keys
#        - demand
#        - matric
#        - all
#    """
#    dvars = demand_vars(states)
#    mvars = matric_vars(react)
#    avars = [dvars, mvars, ['FreshmanEnrolled', 'Numberofapplicantsfulltime']]
#    avars = [el for sublist in avars for el in sublist]
#    #pylint: disable=maybe-no-member
#    avars = np.unique(np.array(avars)).tolist()
#    return {'demand': dvars, 'matric': mvars, 'all': avars}


def reaction_rhs():
    """ Return RHS for reaction function estimation """
    rhs_fs = ['OverallRank_comp']
    rhs_ss = ['OverallRank', 'treat', 'RankTreat']
    rhs_fs = [rhs_ss, rhs_fs]
    rhs_fs = [el for sublist in rhs_fs for el in sublist]
    return {'fs': rhs_fs, 'ss': rhs_ss}


def rank_evol_vars(tilde=0, react='simple', sim=False):
    """ Variables to use for xbeta for ranked or unranked
    Parameters
    ----------
    tilde: for x-tilde vector (no Rank)
    react: reaction variable specification
    sim: if used for simulation routines (True removes "L")
    """
    xtil_vars = reaction_spec(react)[0]
    xtil_vars = [xvar + 'L' for xvar in xtil_vars]
    x_vars = xtil_vars + ['OverallRankL']

    if sim:
        xtil_vars = [el[:-1] for el in xtil_vars]
        x_vars = [el[:-1] for el in x_vars]

    if tilde:
        return xtil_vars
    else:
        return x_vars


def rutil_import():
    """ Import functions from lawrutils into workspace """
    path = dirname(dirname(__file__))
    path = join(path, 'ext', 'lawrutils.R')
    with open(path, 'r') as rutil_f:
        string = ''.join(rutil_f.readlines())
    rutils = STAP(string, 'rutils')
    return rutils






#def nls_fmla(rvar, states):
#    """ Return formula for NLS estimate of reaction functions """
#    # logit link to bound weight in (0, 1)
#    fmla = rvar + ' ~ b0 * OverallRank + b1 * mean + ' \
#           '1/(1 + exp(-b4 * OverallRank))' \
#           ' * (b2 * ' + rvar + 'p + b3 * ' + rvar + 'm) +' \
#           ' (1 - 1/(1 + exp(-b4 * OverallRank))) * ('
#    init = 1
#    for i in range(0, len(states)):
#        if init:
#            add = 'c' + str(i + 5) + ' * ' + states[i]
#            init = 0
#        else:
#            add = ' + c' + str(i + 5) + ' * ' + states[i]
#        fmla = fmla + add
#    fmla = fmla + ')'
#    return fmla
#
#
#def nls_guess(rvar, data_r):
#    """ Return initial guess for reaction functions """
#    guess_fmla = rvar + ' ~ OverallRank + mean + ' + rvar + 'p + ' + \
#                 rvar + 'm + factor(state) - 1'
#    guess_temp = r.coef(r.lm(guess_fmla, data=data_r))
#    guess = [np.array(guess_temp[0:4]).tolist(), [0.00001],
#             np.array(guess_temp[4:]).tolist()]
#    guess = [item for sublist in guess for item in sublist]
#    guess_list = {}
#    for i in range(0, 5):
#        guess_list['b' + str(i)] = guess[i]
#    for i in range(5, len(guess)):
#        guess_list['c' + str(i)] = guess[i]
#    guess_list = ListVector(guess_list)
#    return guess_list
#
#
#def pp_clean(fn):
#    """ Clean autodiff functions for sending to KNITRO """
#    fn = re.sub('TensorConstant{', '', fn)
#    fn = re.sub('}', '', fn)
#    diff = 1
#    left = 0
#    right = 0
#    ip = b.index('f')
#    i = b.index('f')
#    while diff > 0 & i <= len(fn):
#        print '1: %s %s %s %s' % (diff, i, left, right)
#        if fn[i] == '(':
#            left += 1
#            print '2: %s %s %s %s' % (diff, i, left, right)
#        elif fn[i] == ')':
#            right += 1
#            print '3: %s %s %s %s' % (diff, i, left, right)
#        if left:
#            diff = left - right
#        i = i + 1
#        print '4: %s %s %s %s' % (diff, i, left, right)
#    fn = fn[:ip] + fn[(i + 3):]
#    if i == len(fn):
#        print 'Gradient or Hessian incorrectly reformatted'
#    return fn
