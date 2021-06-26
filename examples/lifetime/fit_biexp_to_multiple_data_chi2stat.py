#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Fit simultaneously (global analysis).

Example of fitting a convolved bi-exponential fluorescence
decay simultaneously to two measurements using Chi2 Statistic.

"""

from fluo.fitter import make_global_lifetime_fitter
from matplotlib import pyplot as plt
import numpy as np


def main():
    """Illustrate workflow for fitting a bi-exponential decay.

    Example of fitting a convolved bi-exponential fluorescence
    decay simultaneously to two measurement using Chi2 Statistic.

    """
    file1 = np.loadtxt('../decay_2exp_1ns_02_4ns_08.txt', skiprows=1)
    file2 = np.loadtxt('../decay_2exp_1ns_01_4ns_09.txt', skiprows=1)
    local_times = [file1[:, 0], file2[:, 0]]
    local_decays = [file1[:, 2], file2[:, 2]]
    local_irfs = [file1[:, 1], file2[:, 1]]
    model_kwargs_e2 = {
        'model_components': 2,
        'model_parameters': {
            'amplitude1': {'value': 0.1, 'vary': True, 'min': 1E-6},
            'amplitude2': {'value': 0.1, 'vary': True, 'min': 1E-6},
            'tau1': {'value': 1, 'vary': True, 'min': 1E-6},
            'tau2': {'value': 5, 'vary': True, 'min': 1E-6},
            'offset': {'value': 0.1, 'vary': True},
            'shift': {'value': 0.5, 'vary': True}
        },
        'fit_start': 2,
        'fit_stop': None
    }
    local_model_kwargs_e2 = [
        model_kwargs_e2,
        model_kwargs_e2.copy()
        ]
    # Global convolution fit with Chi2 Statistic
    chi2stat_fitter_global = \
    make_global_lifetime_fitter(
        local_user_kwargs=local_model_kwargs_e2,
        local_times=local_times,
        local_decays=local_decays,
        local_instrument_responses=local_irfs,
        fit_statistic='chi_square_statistic',
        shared=['tau1', 'tau2']
        )
    chi2stat_fit_global = chi2stat_fitter_global.fit()
    local_indexes = chi2stat_fitter_global.local_indexes
    local_best_fits = np.split(
        chi2stat_fit_global.best_fit,
        local_indexes
        ) # split global best_fit into local ones
    # plot
    for ith, (time, decay, irf) in enumerate(zip(local_times,local_decays,local_irfs)):
        plt.plot(time, decay, 'bo', label='decay')
        plt.plot(time, irf, 'go', label='irf')
        plt.plot(
            chi2stat_fitter_global.local_independent_var[ith]['time'],
            local_best_fits[ith],
            'r-',
            label='fit')
        plt.legend(loc='best')
        plt.yscale('log')
        plt.show()

if __name__ == "__main__":
    main()
