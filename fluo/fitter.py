# -*- coding: utf-8 -*-

"""
Module with a Fitter object and factory functions for fitting lifetimes to measurements of fluorescence decay.
"""

import numpy as np
from lmfit import report_fit
import itertools
from .statistics import CStatistic, ChiSquareStatistic, ChiSquareStatisticVariableProjection
from .models import GlobalModel, AddConstant, Linearize, Convolve, Exponential


def iterative_least_squares(FitterClass, iterations):
    """Performs least squares in a loop.

    Performs least squares minimization in iterations, 
    with initial parameters values from previous iteration 
    and variance approximation according to Pearson 
    (based on fitted model).

    Parameters
    ----------
    FitterClass : fluo.Fitter
    iterations : int

    Returns
    -------
    fits : list of lmfit.ModelResult
        List with fit from every iteration.
    """

    print(
        "0-th iteration. Initial fit."
        )
    ini_fit = FitterClass.fit(report=True)
    i_params = ini_fit.params
    fits = [ini_fit]
    for i in range(iterations):
        print()
        print(
            "{}-th iteration".format(i+1)
            )
        FitterClass.statistic = ChiSquareStatistic(variance_approximation='Pearson')
        FitterClass.parameters = i_params
        i_fit = FitterClass.fit(report=True)
        i_params = i_fit.params
        fits.append(i_fit)
    return fits

def make_global_lifetime_fitter(
    local_user_kwargs, 
    local_times, 
    local_decays, 
    local_instrument_responses=None, 
    fit_statistic='c_statistic', 
    shared=None):
    """Makes a fitter for simultaneous (global) fitting.

    Makes `fluo.Fitter` object for simultaneous (global) fitting multiple measurements.

    Parameters
    ----------
    local_user_kwargs : list of dict
        List of dict with user provided info about model and fit.
    local_times : list of ndarray
        List of 1D ndarray with times (x-scale of data).
    local_decays : list of ndarray
        List of 1D ndarray with fluorescence decays (y-scale of data).
    local_instrument_responses : list of ndarray, optional
        List of 1D ndarray with instrument_response functions 
        (for convolution with calculated model).
    fit_statistic : str, optional
        Statisic used in fitting minimization. 
        Accepts the following str: 'c_statistic', 'chi_square_statistic', 
        'chi_square_statistic_variable_projection'.
    shared : list of str, optional
        List of parameters names shared between fitted measurements.

    Raises
    ------
    ValueError
        If invalid `fit_statistic` is provided.

    Returns
    -------
    fluo.Fitter
    """
    if local_instrument_responses is None:
        local_instrument_responses = iter([])
    local_zipped = itertools.zip_longest(
        local_user_kwargs, 
        local_times, 
        local_decays, 
        local_instrument_responses
        )
    local_fitter_classes = [
        make_lifetime_fitter(*args) for args in local_zipped
    ]

    global_pre_fitter_cls = GlobalModel(
        FitterClasses=local_fitter_classes, 
        shared=shared)
    independent_var = dict(
        independent_var = global_pre_fitter_cls.local_independent_var
        )
    dependent_var = np.concatenate(global_pre_fitter_cls.local_dependent_var)
    statistic_cls = global_pre_fitter_cls.statistic

    return Fitter(
            ModelClass=global_pre_fitter_cls, 
            independent_var=independent_var, 
            dependent_var=dependent_var,
            statistic=statistic_cls
    )


def make_lifetime_fitter(
    user_kwargs, 
    time, 
    decay, 
    instrument_response=None, 
    fit_statistic='c_statistic'):
    """Makes a fitter.

    Makes `fluo.Fitter` object for fitting a single measurement.

    Parameters
    ----------
    user_kwargs : dict
        Dict with user provided info about model and fit.
    time : ndarray
        1D ndarray with times (x-scale of data).
    decay : ndarray
        1D ndarray with fluorescence decays (y-scale of data).
    instrument_response : ndarray, optional
        1D ndarray with instrument_response functions 
        (for convolution with calculated model).
    fit_statistic : str, optional
        Statisic used in fitting minimization. 
        Accepts the following str: 'c_statistic', 'chi_square_statistic', 
        'chi_square_statistic_variable_projection'.

    Raises
    ------
    ValueError
        If invalid `fit_statistic` is provided.

    Returns
    -------
    fluo.Fitter
    """
    
    allowed_fit_statistics = dict(
    c_statistic = CStatistic(),
    chi_square_statistic = ChiSquareStatistic(),
    chi_square_statistic_variable_projection = ChiSquareStatisticVariableProjection()
    )

    try:
        statistic_cls = allowed_fit_statistics[fit_statistic]
    except KeyError:
        allowed_fit_statistics_names = ", ".join(list(allowed_fit_statistics.keys()))
        raise ValueError(
            "fit_statistic: '{0}' not implemented. Available fit_statistic: {1}".format(
                fit_statistic, 
                allowed_fit_statistics_names)
        )

    # pre-process fit range
    user_kwargs = user_kwargs.copy()
    fit_start, fit_stop = user_kwargs.pop('fit_start'), user_kwargs.pop('fit_stop')
    if fit_start is None:
        fit_start = 0
    if fit_stop is None:
        fit_stop = np.inf
    range_mask = (time >= fit_start) & (time <= fit_stop)
    decay = decay[range_mask].astype(float)
    time = time[range_mask].astype(float)

    if instrument_response is None:
        exponential_cls = Exponential(**user_kwargs)
        independent_var = dict(
            time=time
        )
    else:
        exponential_cls = Convolve(Exponential(**user_kwargs))
        instrument_response = instrument_response[range_mask].astype(float)
        independent_var = dict(
            time=time, 
            instrument_response=instrument_response
            )

    if isinstance(
        statistic_cls, 
        ChiSquareStatisticVariableProjection):  
        # pre_fitter_cls = Linear(user_kwargs)
        #     independent_var = dict(
        #         independent_var = 
        #     )
        #     statistic_cls = ChiSquareStatistic()      
        return Fitter(
            ModelClass=exponential_cls,
            independent_var=independent_var, 
            dependent_var=decay,
            statistic=statistic_cls
            )
    else:
        return Fitter(
            ModelClass=AddConstant(Linearize(exponential_cls)),
            independent_var=independent_var, 
            dependent_var=decay,
            statistic=statistic_cls
        )


class Fitter(): 
    """Fitter object for fitting.

    Parameters
    ----------
    ModelClass : fluo.Model
        Model class inheriting from fluo.Model
    independent_var : dict
        Independent variables for a model evaluation. Dict with names of independent variables encoded by keys (str)
        and values as ndarrays.
    dependent_var : ndarray
        1D ndarray with dependent variable for fitting.
    statistic : fluo.Statistic
        Statistic class for fitting.

    Attributes
    ----------
    parameters : lmfit.Parameters
        lmfit.Parameters for model evaluation.
    model : fluo.GenericModel

    Methods
    -------
    fit : lmfit.ModelResult
    """

    def __init__(self, ModelClass, independent_var, dependent_var, statistic):
        self.ModelClass = ModelClass
        self.independent_var = independent_var
        self.dependent_var = dependent_var
        self.statistic = statistic 
        self.parameters = ModelClass.make_parameters()
        self.model = ModelClass.make_model(**independent_var)
                
    def fit(self, report=True):
        """Performes a fit.

        Parameters
        ----------
        report : bool, optional
            Report fit (True by default).

        Returns
        -------
        lmfit.ModelResult

        """
        self.name = '{} fitted using {}'.format(
            self.model.name,
            self.statistic.name)
        result = self.model.generic_fit(
                    data=self.dependent_var,
                    statistic=self.statistic,
                    params=self.parameters
                    )
        if report:
            print('Report: {}'.format(self.name))
            report_fit(result)
        return result


def autocorrelation(residuals):
    """Calculates residuals autocorrelation.

    Calculates correlation between residuals in i-th and (i+j)-th channels.

    Parameters
    ----------
    residuals : ndarray

    Returns
    -------
    ndarray
    """

    residuals_full = residuals
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    inv_n = 1. / n
    denominator = inv_n * np.sum(np.square(
        residuals))  # normalization weight in autocorrelation function
    residuals = list(residuals)
    m = n // 2
    numerator = []
    for j in range(m):
        k = n - j
        numerator_sum = 0.0
        for i in range(k):
            numerator_sum += residuals[i] * residuals[i + j]
        numerator.append(numerator_sum / k)
    numerator = np.array(numerator)
    autocorr = numerator / denominator
    over_range = np.array([np.nan] * len(residuals_full))
    autocorr = np.append(autocorr, over_range)
    return autocorr