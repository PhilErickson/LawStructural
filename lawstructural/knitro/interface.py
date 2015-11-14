# pylint: disable=wildcard-import, unused-wildcard-import, too-many-arguments
"""
An interface with the Ziena KNITRO solver. Requires an active KNITRO license.
"""

from __future__ import print_function, division
import lawstructural.knitro.knitro as ktr
from lawstructural.knitro.knitroNumPy import *
from lawstructural.knitro.knitro import *

# ----------------------------------------------------------------------------
# Default solver options and ktrsolve function definition


def ktrsolve(fun, guess, **kwargs):
    """
    Interface for Ziena KNITRO solver.

    Parameters
    ----------
    fun : callable
        Objective function.
    guess : ndarray
        Initial guess.
    constr : callable, optional
        Constraint vector; Default is None.
    grad : callable, optional
        Definition of function gradient.
    jac : callable, optional
        Definition of constraint jacobian.
    hess : callable, optional
        Definition of the function/constraint matrix Hessian.
    ktr_defs : dict, optional
        Optimization problem definitions; Default set by
        self._ktr_defs_default(). Has keys

            - 'obj_goal' : KTR_OBJGOAL_* (e.g. MINIMIZE)
            - 'obj_type' : KTR_OBJTYPE_* (e.g. GENERAL)
            - 'bnds_lo' : ndarray, shape=[n, ]
            - 'bnds_up' : ndarray, shape=[n, ]
            - 'm_constr' : number of constraints,
            - 'c_type' : KTR_CONTYPE_* (e.g. QUADRATIC)
            - 'c_bnds_lo' : ndarray, shape=[m, ]
            - 'c_bnds_up' : ndarray, shape=[m, ]}
        with 'n' the number of parameters to be estimated and 'm' the number
        of constraints
    sparse_jac : ndarray, optional
        Sparsity pattern of constraint Jacobian matrix. Should have dimensions
        [m_constr x m_constr] with

        .. math::

            sparse\_jac[i, j] =
                \\begin{cases}
                    0 \\text{ if } Jac[i, j] = 0 \\\\
                    1 \\text{ o.w.}
                \\end{cases}

        Default is non-zero for all elements.

        .. math::

           (a + b)^2  &=  (a + b)(a + b) \\
                      &=  a^2 + 2ab + b^2

    sparse_hess : ndarray, optional
        Sparsity pattern of Hessian matrix. Should have dimensions
        [m_constr x m_constr] and has elements similarly defined as sparse_jac.
        Default is non-zero for upper diagonal elements (by symmentry, implying
        non-zero for all elements).
    options : dict, optional
        Options for solver. Key/value pair corresponds to KNITRO option
        name/value. For example,

        {'outlev': 'all',
         'gradopt': 3,
         'hessopt': ktr.KTR_HESSOPT_BFGS,
         'feastol': 1.0E-10}

        See `KNITRO manual <http://www.ziena.com/documentation.htm>`_ for
        more options.
    verbose : Binary, optional
        If True (default), show final status of solver

    Notes
    -----
    This function requires an active KNITRO license to work.

    """
    ktr_int = Interface(fun, guess, **kwargs)
    if 'verbose' in kwargs:
        return ktr_int.solve(kwargs['verbose'])
    else:
        return ktr_int.solve()


# ----------------------------------------------------------------------------
# Interface class

class Interface(object):
    """ Interface class for Ziena KNITRO solver. This need not be called
    independently of ktrsolve and has the same parameters.

    """

    def __init__(self, fun, guess, **kwargs):
        self.fun = fun
        self.guess = guess
        self.kwargs = kwargs
        self.ktr_defs = self._ktr_defs_set()
        self.options = self._options_set()
        self.out = {}
        self.ktr_current = None

    def _options_default(self):
        """ Default solver options for Interface """
        if 'grad' in self.kwargs:
            gradopt = 1
        else:
            gradopt = 3

        if 'hess' in self.kwargs:
            hessopt = 1
        else:
            hessopt = 2

        return {'outlev': 'all',
                'gradopt': gradopt,
                'hessopt': hessopt,
                'feastol': 1.0E-10,
                'maxit': 500}

    def _ktr_defs_default(self):
        """ Default problem definitions for unconstrained problem
        Returns:
            dictionary with ktr_defs defaults
        """
        out = {}
        out['n_params'] = len(self.guess)
        out['obj_goal'] = ktr.KTR_OBJGOAL_MINIMIZE
        out['obj_type'] = ktr.KTR_OBJTYPE_GENERAL
        out['bnds_lo'] = np.array([-ktr.KTR_INFBOUND]).repeat(out['n_params'])
        out['bnds_up'] = np.array([ktr.KTR_INFBOUND]).repeat(out['n_params'])
        if 'constr' in self.kwargs:
            out['m_constr'] = len(self.kwargs['constr'](self.guess))
            out['c_type'] = np.array([ktr.KTR_CONTYPE_GENERAL]).\
                            repeat(out['m_constr'])
            out['c_bnds_lo'] = np.array([-ktr.KTR_INFBOUND]).\
                               repeat(out['m_constr'])
            out['c_bnds_up'] = np.array([ktr.KTR_INFBOUND]).\
                               repeat(out['m_constr'])

            if 'sparse_jac' in self.kwargs:
                sparse_jac = self.kwargs['sparse_jac']
            else:
                sparse_jac = np.ones((out['m_constr'], out['n_params']))

            if 'sparse_hess' in self.kwargs:
                sparse_hess = self.kwargs['sparse_hess']
            else:
                sparse_hess = np.triu(np.ones((out['n_params'],
                                               out['n_params'])))

            (out['jac_ix_constr'], out['jac_ix_var']) = \
                np.where(sparse_jac == 1)
            (out['hess_row'], out['hess_col']) = \
                np.where(np.triu(sparse_hess) == 1)
        else:
            out['m_constr'] = 0
            out['c_type'] = None
            out['c_bnds_lo'] = None
            out['c_bnds_up'] = None
            (out['jac_ix_constr'], out['jac_ix_var']) = (None, None)
            (out['hess_row'], out['hess_col']) = (None, None)

        return out

    def _ktr_defs_set(self):
        """ Set values for ktr_def based on defaults and user input """
        ktr_defs = self._ktr_defs_default()
        if 'ktr_defs' in self.kwargs:
            for key in self.kwargs['ktr_defs']:
                try:
                    ktr_defs[key] = self.kwargs['ktr_defs'][key]
                except:
                    raise RuntimeError("Error setting problem definition '" +
                                       key + "'")
        return ktr_defs

    def _options_set(self):
        """ Set options for ktr_def based on defaults and user input """
        options = self._options_default()
        if 'options' in self.kwargs:
            for key in self.kwargs['options']:
                try:
                    options[key] = self.kwargs['options'][key]
                except:
                    raise RuntimeError("Error setting problem option '" +
                                       key + "'")
        return options

    def _param_set(self):
        """ Set parameters for current KNITRO instance """
        for name, option in self.options.iteritems():
            if isinstance(option, str):
                if ktr.KTR_set_char_param_by_name(self.ktr_current, name,
                                                  option):
                    raise RuntimeError("Error setting parameter '" +
                                       name + "'")

            elif isinstance(option, int):
                if ktr.KTR_set_int_param_by_name(self.ktr_current, name,
                                                 option):
                    raise RuntimeError("Error setting parameter '" +
                                       name + "'")

            elif isinstance(option, float):
                if ktr.KTR_set_double_param_by_name(self.ktr_current, name,
                                                    option):
                    raise RuntimeError("Error setting parameter '" +
                                       name + "'")

            else:
                raise RuntimeError("Option format not recognized.")

    def _eval_fc(self, coef_out, constr):
        """ KNITRO-formatted function with constraint """
        obj = self.fun(coef_out)
        if 'constr' in self.kwargs:
            constr[0:self.ktr_defs['m_constr']] = \
                self.kwargs['constr'](coef_out)
        else:
            const = None

        return obj

    def _eval_ga(self, coef_out, obj_grad, jac):
        """ KNITRO-formatted first derivatives for function and constraints """
        if 'grad' not in self.kwargs:
            raise RuntimeError("No function gradient supplied.")
        if 'jac' not in self.kwargs:
            raise RuntimeError("No constraint jacobian supplied.")

        obj_grad[0:self.ktr_defs['n_params']] = self.kwargs['grad'](coef_out)

        if 'constr' in self.kwargs:
            try:
                n_jac = (len(self.ktr_defs['jac_ix_constr']) *
                         len(self.ktr_defs['jac_ix_var']))
            except:
                raise RuntimeError("Invalid input for argument 'sparse_jac'.")
            jac[0:n_jac] = self.kwargs['jac'](coef_out)
        else:
            jac = None

    def _eval_h(self, coef_out, lambda_, sigma, hess):
        """ KNITRO-formatted Hessian of function and constraints """
        if 'hess' not in self.kwargs:
            raise RuntimeError("No function gradient supplied.")

        try:
            n_hess = (len(self.ktr_defs['hess_row']) *
                      len(self.ktr_defs['hess_col']))
        except:
            raise RuntimeError("Invalid input for argument 'sparse_hess'.")

        hess[0:n_hess] = self.kwargs['hess'](coef_out, lambda_, sigma)

    def _callback_eval_fc(self, eval_request_code, n_params, m_constr, nnz_j,
                          nnz_h, coef_out, lambda_, obj, constr, obj_grad,
                          jac, hessian, hess_vector, user_params):
        """ Function callback for self.fun """
        if eval_request_code == ktr.KTR_RC_EVALFC:
            obj[0] = self._eval_fc(coef_out, constr)
            return 0
        else:
            return ktr.KTR_RC_CALLBACK_ERR

    def _callback_eval_ga(self, eval_request_code, n_params, m_constr, nnz_j,
                          nnz_h, coef_out, lambda_, obj, const, obj_grad, jac,
                          hessian, hess_vector, user_params):
        """ Function callback for gradient """
        if eval_request_code == ktr.KTR_RC_EVALGA:
            self._eval_ga(coef_out, obj_grad, jac)
            return 0
        else:
            #raise RuntimeError("This was the problem!")
            print("Eval Request Code:", eval_request_code)
            return ktr.KTR_RC_CALLBACK_ERR

    def _callback_eval_h(self, eval_request_code, n_params, m_constr, nnz_j,
                         nnz_h, coef_out, lambda_, obj, const, obj_grad, jac,
                         hessian, hess_vector, user_params):
        """ Function callback for Hessian """
        if eval_request_code == ktr.KTR_RC_EVALH:
            self._eval_h(coef_out, lambda_, 1.0, hessian)
            return 0

        elif eval_request_code == ktr.KTR_RC_EVALH_NO_F:
            self._eval_h(coef_out, lambda_, 0.0, hessian)
            return 0

        else:
            return ktr.KTR_RC_CALLBACK_ERR

    def _callback_reg(self):
        """ Register function, gradient, and Hessian callbacks """
        if ktr.KTR_set_func_callback(self.ktr_current, self._callback_eval_fc):
            raise RuntimeError("Error registering function callback.")

        if 'grad' in self.kwargs:
            if ktr.KTR_set_grad_callback(self.ktr_current,
                                         self._callback_eval_ga):
                raise RuntimeError("Error registering gradient callback.")

        if 'hess' in self.kwargs:
            if ktr.KTR_set_hess_callback(self.ktr_current,
                                         self._callback_eval_h):
                raise RuntimeError("Error registering Hessian callback.")

    def _out_init(self):
        """ Initialize self.out """
        self.out['coef'] = np.zeros(self.ktr_defs['n_params'])
        self.out['lambda_'] = np.zeros(self.ktr_defs['m_constr'] + \
                              self.ktr_defs['n_params'])
        self.out['obj'] = np.array([0])

    def _init_problem(self):
        """ Initialize KNITRO problem """
        return ktr.KTR_init_problem(self.ktr_current,
                                    self.ktr_defs['n_params'],
                                    self.ktr_defs['obj_goal'],
                                    self.ktr_defs['obj_type'],
                                    self.ktr_defs['bnds_lo'],
                                    self.ktr_defs['bnds_up'],
                                    self.ktr_defs['c_type'],
                                    self.ktr_defs['c_bnds_lo'],
                                    self.ktr_defs['c_bnds_up'],
                                    self.ktr_defs['jac_ix_var'],
                                    self.ktr_defs['jac_ix_constr'],
                                    self.ktr_defs['hess_row'],
                                    self.ktr_defs['hess_col'],
                                    self.guess, None)

    def solve(self, verbose=True):
        """
        Routine for calling the knitro solver.

        Parameters
        ----------
        verbose : boolean, optional
            If true, displays basic convergence results. Default is True.

        Returns
        -------
        out : dict
            Dictionary of self.solve() output with keys

                - 'coef': estimated parameters
                - 'lambda_': lagrange multiplier
                - 'obj': objective evaluated at estimated parameters
        """
        self.ktr_current = ktr.KTR_new()
        if self.ktr_current == None:
            raise RuntimeError("Failed to find a Ziena license.")

        # Set KNITRO parameters and register Callbacks
        self._param_set()
        self._callback_reg()

        # Initialize KNITRO with problem definition
        ret = self._init_problem()
        if ret:
            raise RuntimeError("Error initializing the problem, KNITRO "
                               "status = %d" % ret)

        # Initialize output dictionary and solve
        self._out_init()
        n_status = ktr.KTR_solve(self.ktr_current, self.out['coef'],
                                 self.out['lambda_'], 0,
                                 self.out['obj'], None, None, None, None,
                                 None, None)
        if n_status != 0:
            if n_status == -100:
                print("Approximate solution acheived, final status = %d" %
                      n_status)
            else:
                raise RuntimeError("KNITRO failed to solve the problem, final "
                                   "status = %d" % n_status)
        else:
            if verbose:
                print("KNITRO successful, feasibility violation    "
                      "= %e" % ktr.KTR_get_abs_feas_error(self.ktr_current))
                print("                   KKT optimality violation "
                      "= %e" % ktr.KTR_get_abs_opt_error(self.ktr_current))
        ktr.KTR_free(self.ktr_current)
        return self.out
