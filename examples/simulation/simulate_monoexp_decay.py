#!/usr/bin/env python3

# -*- coding: utf-8 -*-
""""Simulate a mono-exponential decay.

Example of simulating a mono-exponential fluorescence decay with Monte Carlo method.

"""

from fluo.simulation import make_simulation
from matplotlib import pyplot as plt
import numpy as np


def main():
    """Illustrate workflow to simulate a mono-exponential decay.

    Example of simulating a convolved mono-exponential fluorescence
    decay. The convolution is calculated using Monte Carlo method.

    """
    file = np.loadtxt('../irf.txt', skiprows=1)
    time, irf = file[:, 0], file[:, 1]
    model_kwargs_e1 = {
        'model_components': 1,
        'model_parameters': {
            'amplitude1': {'value': 7000},
            'offset': {'value': 0.1},
            'tau1': {'value': 5},
            'shift': {'value': 0.5}
        }
    }
    # simulate
    simulation_1exp_5ns = \
    make_simulation(model_kwargs_e1, time, irf, verbose=True)
    # save & plot
    np.savetxt(
        '../decay_1exp_5ns.txt',
        np.stack((time, irf, simulation_1exp_5ns), axis=1),
        delimiter='\t',
        header='time\tirf\tsimulation'
        )
    plt.plot(simulation_1exp_5ns)
    plt.yscale('log')
    plt.show()


if __name__ == "__main__":
    main()
