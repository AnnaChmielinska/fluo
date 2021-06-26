#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""Tail-fit a mono-exponential decay.

Example of tail-fitting a mono-exponential fluorescence
decay to a single measurement using C Statistic.

"""

from fluo.fitter import make_lifetime_fitter
from matplotlib import pyplot as plt
import numpy as np


def main():
    """Illustrate workflow for tail-fitting a mono-exponential decay.

    Example of tail-fitting a mono-exponential fluorescence
    decay to a single measurement using C Statistic.

    """
    file = np.loadtxt('../decay_1exp_5ns.txt', skiprows=1)
    time, irf, decay = file[:, 0], file[:, 1], file[:, 2]
    model_kwargs_e1_tail = {
        'model_components': 1,
        'model_parameters': {
            'amplitude1': {'value': 7000, 'vary': True, 'min': 1E-6},
            'offset': {'value': 0.1, 'vary': True, 'min': 1E-6},
            'tau1': {'value': 5, 'vary': True, 'min': 1E-6},
        },
        'fit_start': 12,
        'fit_stop': None
    }
    # Tail fit with Chi2 Statistic
    cstat_tail_fitter = \
    make_lifetime_fitter(
        model_kwargs_e1_tail,
        time,
        decay,
        fit_statistic='c_statistic'
        )
    cstat_tail_fit = cstat_tail_fitter.fit(report=True)
    # plot
    plt.plot(time, decay, 'bo', label='decay')
    plt.plot(
        cstat_tail_fitter.independent_var['time'],
        cstat_tail_fit.best_fit,
        'r-',
            label='fit')
    plt.legend(loc='best')
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()
