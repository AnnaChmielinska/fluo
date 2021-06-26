# -*- coding: utf-8 -*-

"""Provide the Statistic object.

Statistic provides definition of objective function and optimization method for fitting process.

"""

from abc import ABCMeta, abstractmethod
import warnings
import numpy as np
import scipy
import scipy.linalg

class Statistic():
    """Abstract class for the Statistic object.

    Parameters
    ----------
    name : str
    optimization_method : str
        One of: 'nelder', 'powell', 'leastsq'.

    Methods
    -------
    objective_func : ndarray

    """

    __metaclass__ = ABCMeta

    def __init__(self, name, optimization_method):
        """Initialize a Statistic object.

        Parameters
        ----------
        name : str
        optimization_method : str
            One of: 'nelder', 'powell', 'leastsq'.

        """
        self.name = name
        self.optimization_method = optimization_method

    @abstractmethod
    def objective_func(self, *args):
        """Objective function for optimization."""
        raise NotImplementedError("`Statistic.objective_func` not implemented")


class CStatistic(Statistic):
    """Class for the C statistic.

    C statistic for data with Poisson distribution.
    The objective funtion is calculated as log-likelihood of the product of
    Poisson probabilities in each bin.

    Parameters
    ----------
    optimization_method : str, optional
        'powell' by default. Accepts the following str: 'nelder', 'powell'.

    Attributes
    ----------
    name : str

    Methods
    -------
    objective_func : ndarray

    """

    _allowed_optimization_methods = ['nelder', 'powell']

    def __init__(self, optimization_method='powell'):
        """Initialize the Statistic object.

        Parameters
        ----------
        optimization_method : str, optional
            'powell' by default. Accepts the following str: 'nelder', 'powell'.

        """
        if optimization_method not in self._allowed_optimization_methods:
            raise ValueError(
                "Only {} are allowed as a `optimization_method`.".format(
                    ', '.join(['`'+meth+'`' for meth \
                    in self._allowed_optimization_methods])))
        super().__init__(
            name='c_statistic',
            optimization_method=optimization_method
        )

    def objective_func(self, model, dependent_var):
        """Objective function minimized in fit.

        Parameters
        ----------
        model : ndarray
            Values in model must not be negative. The negative values are
            dropped from model.
        dependent_var : ndarray
            Values in dependent_var must not equal zero. The zero-valued
            entries are dropped from dependent_var.

        Raises
        ------
        warnings.warn
            If negative values are encountered in model.

        Returns
        -------
        ndarray

        """
        model_copy = np.copy(model)
        dependent_var_copy = np.copy(dependent_var)
        if np.sum(model < 0):
            warnings.warn("""
            Negative values in model.
            Make sure the parameters are constrained in such a way that the model is not negative-valued anywhere.
            Dropping negative values from the model
            """)
            model_copy[model < 0] = np.nan
        dependent_var_copy[dependent_var == 0] = np.nan
        result = dependent_var_copy * np.log(model_copy / dependent_var_copy)
        res = -2*(result + (dependent_var_copy - model_copy))
        return np.nan_to_num(res)


class ChiSquareStatistic(Statistic):
    """Class for Chi square statistic.

    Chi square statistic for data with Gaussian distribution.
    The objective funtion is calculated according to the
    weighted least squares. The weights are calculated according
    to Neyman or Pearson variance approximation.

    Parameters
    ----------
    optimization_method : str, optional
        'leastsq' by default.
    variance_approximation : str, optional
        'Neyman' by default. Accepts the following str: 'Neyman', 'Pearson', None.

    Attributes
    ----------
    name : str

    Methods
    -------
    objective_func : ndarray
    weight : ndarray
    apply_weight : ndarray

    """

    _allowed_variance_approximations = [None, 'Neyman', 'Pearson']

    def __init__(self, optimization_method='leastsq',
                 variance_approximation='Neyman'):
        """Initialize Model object.

        Parameters
        ----------
        optimization_method : str, optional
            'leastsq' by default.
        variance_approximation : str, optional
            'Neyman' by default. Accepts the following str: 'Neyman',
            'Pearson', None.
        """
        if variance_approximation not in self._allowed_variance_approximations:
            raise ValueError(
                "Only {} are allowed as a `variance_approximation`.".format(
                    ', '.join(['`'+appr+'`' for \
                    appr in self._allowed_variance_approximations])))
        self.variance_approximation = variance_approximation
        super().__init__(
            name='chi_square_statistic with {} variance_approximation'.format(
                self.variance_approximation
                ),
            optimization_method=optimization_method)

    def objective_func(self, model, dependent_var):
        """Objective function minimized in fit.

        Calculates weighted residuals (difference between provided model and
        dependent variable). The weights are calculated according to
        `variance_approximation`.

        Parameters
        ----------
        model : ndarray
        dependent_var : ndarray

        Returns
        -------
        ndarray

        """
        if self.variance_approximation is not None:
            model, dependent_var = self._weight_input(model, dependent_var)
        return model - dependent_var

    def _weight_input(self, model, dependent_var):
        if self.variance_approximation == 'Neyman':
            weights = self.weight(dependent_var)
        elif self.variance_approximation == 'Pearson':
            weights = self.weight(model)
        return (self.apply_weight(model, weights),
                self.apply_weight(dependent_var, weights))

    @staticmethod
    def weight(array):
        """Calculate reciprocal square root.

        Parameters
        ----------
        array : ndarray

        Returns
        -------
        ndarray

        """
        array[array <= 0] = 1.
        return np.reciprocal(np.sqrt(array))

    @staticmethod
    def apply_weight(array, weights):
        """Multiply `array` by `weights` row-wise.

        Parameters
        ----------
        array : ndarray
        weights : ndarray

        Returns
        -------
        ndarray
            Weighted array.

        """
        ncols, *nrows = array.shape
        arr_weighted = np.copy(array)
        try:
            arr_weighted *= np.tile(weights, (*nrows, 1)).T
        except ValueError:
            arr_weighted *= weights

        return arr_weighted


class ChiSquareStatisticVariableProjection(ChiSquareStatistic):
    """Class for Chi square statistic for variable projection.

    Chi square statistic for data with Gaussian distribution.
    The objective funtion is calculated according to the
    weighted separable non-linear least squares, where linear parameters are
    eliminated. The weights are calculated according to Neyman or Pearson
    variance approximation.

    Parameters
    ----------
    optimization_method : str, optional
        'leastsq' by default.
    variance_approximation : str, optional
        'Neyman' by default. Accepts the following str: 'Neyman', 'Pearson',
        None.

    Attributes
    ----------
    name : str

    Methods
    -------
    objective_func : ndarray
    weight : ndarray
    apply_weight : ndarray

    """

    def __init__(self, optimization_method='leastsq',
                 variance_approximation='Neyman'):
        """Initialize Model object.

        Parameters
        ----------
        optimization_method : str, optional
            'leastsq' by default.
        variance_approximation : str, optional
            'Neyman' by default. Accepts the following str: 'Neyman', 'Pearson',
            None.

        """
        super().__init__(
            optimization_method=optimization_method,
            variance_approximation=variance_approximation)
        self.name = 'chi_square_statistic_variable_projection'

    def objective_func(self, model, dependent_var):
        """Objective function minimized in fit.

        Calculates weighted residuals as varible projection of `dependent_var`.
        The weights are calculated according to `variance_approximation`.

        Parameters
        ----------
        model : ndarray
        dependent_var : ndarray

        Returns
        -------
        ndarray

        """
        if self.variance_approximation is not None:
            model, dependent_var = self._weight_input(model, dependent_var)
        pseudoinv_model = scipy.linalg.pinv(model)
        model = model.dot(pseudoinv_model.dot(dependent_var))
        return model - dependent_var
