#!/usr/bin/env python3

# -*- coding: utf-8 -*-
""""Simulate a bi-exponential decay.

Example of simulating a bi-exponential fluorescence decay with Monte Carlo method.

"""

from fluo.simulation import make_simulation
from matplotlib import pyplot as plt
import numpy as np


def main():
    """Illustrate workflow to simulate a bi-exponential decay.

    Example of simulating a convolved bi-exponential fluorescence
    decay. The convolution is calculated using Monte Carlo method.

    """
    file = np.loadtxt('../irf.txt', skiprows=1)
    time, irf = file[:, 0], file[:, 1]
    model_kwargs_e2 = {
        'model_components': 2,
        'model_parameters': {
            'amplitude1': {'value': 0.1},
            'amplitude2': {'value': 0.9},
            'tau1': {'value': 1},
            'tau2': {'value': 4},
            'shift': {'value': 0.5},
            'offset': {'value': 0.1}
        }
    }
    # simulate
    simulation_2exp_1ns_01_4ns_09 = \
    make_simulation(model_kwargs_e2, time, irf, verbose=True)
    # save & plot
    np.savetxt(
        '../decay_2exp_1ns_01_4ns_09.txt',
        np.stack((time, irf, simulation_2exp_1ns_01_4ns_09), axis=1),
        delimiter='\t',
        header='time\tirf\tsimulation'
        )
    plt.plot(simulation_2exp_1ns_01_4ns_09)
    plt.yscale('log')


if __name__ == "__main__":
    main()
