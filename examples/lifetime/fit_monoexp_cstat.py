#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Fit a mono-exponential decay.

Example of fitting a convolved mono-exponential fluorescence
decay to a single measurement using C Statistic.

"""

from fluo.fitter import make_lifetime_fitter
from matplotlib import pyplot as plt
import numpy as np


def main():
    """Illustrate workflow for fitting a mono-exponential decay.

    Example of fitting a convolved mono-exponential fluorescence
    decay to a single measurement using C Statistic.

    """
    file = np.loadtxt('../decay_1exp_5ns.txt', skiprows=1)
    time, irf, decay = file[:, 0], file[:, 1], file[:, 2]
    model_kwargs_e1 = {
        'model_components': 1,
        'model_parameters': {
            'amplitude1': {'value': 0.06, 'vary': True},
            'offset': {'value': 0.1, 'vary': True},
            'tau1': {'value': 5, 'vary': True},
        },
        'fit_start': 2,
        'fit_stop': None
    }
    # Convolution fit with Chi2 Statistic
    cstat_fitter = \
    make_lifetime_fitter(
        model_kwargs_e1,
        time,
        decay,
        instrument_response=irf,
        fit_statistic='c_statistic'
        )
    cstat_fit = cstat_fitter.fit(report=True)
    # exit(1)
    # plot
    plt.plot(time, decay, 'bo', label='decay')
    plt.plot(time, irf, 'go', label='irf')
    plt.plot(
        cstat_fitter.independent_var['time'],
        cstat_fit.best_fit,
        'r-',
        label='fit')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()
