#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Iterative least squares.

Example of fitting a convolved bi-exponential fluorescence decay to single
measurements of using iterative least squares. This method should elimate
the bias introduced by fitting using Gaussian distribution. For a comparison,
the fit using C Statistic is included.

"""

from fluo.fitter import make_lifetime_fitter, iterative_least_squares
from matplotlib import pyplot as plt
import numpy as np
CMAP = plt.get_cmap('gist_rainbow')


def main():
    """Illustrate workflow for fitting a bi-exponential decay.

    Example of fitting a convolved bi-exponential fluorescence
    decay to a single measurement using iterative least squares
    with Chi2 Statistic and Pearson variance approximation.

    """
    file = np.loadtxt('../decay_2exp_1ns_02_4ns_08.txt', skiprows=1)
    time, irf, decay = file[:, 0], file[:, 1], file[:, 2]
    model_kwargs_e2 = {
        'model_components': 2,
        'model_parameters': {
            'amplitude1': {'value': 0.5, 'vary': True, 'min': 1E-6},
            'amplitude2': {'value': 0.5, 'vary': True, 'min': 1E-6},
            'tau1': {'value': 1, 'vary': True, 'min': 1E-6},
            'tau2': {'value': 5, 'vary': True, 'min': 1E-6},
            'offset': {'value': 0.1, 'vary': True},
            'shift': {'value': 1, 'vary': True}
        },
        'fit_start': 2,
        'fit_stop': None
    }
    # ConvolutionFitter C Statistic for comparison
    model_kwargs_e2_cstat = {
        'model_components': 2,
        'model_parameters': {
            'amplitude1': {'value': 0.5, 'vary': True, 'min': 1E-6},
            'amplitude2': {'value': 0.5, 'vary': True, 'min': 1E-6},
            'tau1': {'value': 1, 'vary': True, 'min': 1E-6},
            'tau2': {'value': 5, 'vary': True, 'min': 1E-6},
            'offset': {'value': 0.1, 'vary': True, 'min': 1E-6},
            'shift': {'value': 1, 'vary': True}
        },
        'fit_start': 2,
        'fit_stop': None
    }
    cstat_fitter = \
    make_lifetime_fitter(
        model_kwargs_e2_cstat,
        time,
        decay,
        instrument_response=irf,
        fit_statistic='c_statistic'
        )
    cstat_fit = cstat_fitter.fit()
    # ConvolutionFitter Chi2 Statistic for initialization
    chi2stat_fitter = \
    make_lifetime_fitter(
        model_kwargs_e2,
        time,
        decay,
        instrument_response=irf,
        fit_statistic='chi_square_statistic'
        )
    n_iter = 5
    fits = iterative_least_squares(chi2stat_fitter, n_iter)
    # plot
    plt.plot(time, decay, 'bo', label='decay')
    plt.plot(time, irf, 'go', label='irf')
    plt.plot(
        cstat_fitter.independent_var['time'],
        cstat_fit.best_fit,
        label='C stat',
        color='k'
        )
    for ith, fit in enumerate(fits):
        plt.plot(
            chi2stat_fitter.independent_var['time'],
            fit.best_fit,
            label='{}-ith iter'.format(ith),
            color=CMAP(ith / n_iter)
            )
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()
