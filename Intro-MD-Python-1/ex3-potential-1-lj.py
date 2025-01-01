#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#                                                                              #
##################### Author: Y-L WANG & Date: 2024-Feb-11 #####################

import numpy as np
import matplotlib.pyplot as plt

epsilon = 1.0  # energy in units of k_BT
sigma = 1.0    # distance in units of sigma


def lj_potential(r, epsilon, sigma):
    energy_replusion = 4.0*epsilon*(sigma/r)**12
    energy_attraction = -4.0*epsilon*(sigma/r)**6
    energy = energy_replusion - energy_attraction
    energy = 4.0*epsilon*((sigma/r)**12 - (sigma/r)**6)
    force_repulsion = 48.0*epsilon*np.power(sigma, 12)/np.power(r, 13) 
    force_attraction = -24.0*epsilon*np.power(sigma, 6)/np.power(r, 7)
    force = force_repulsion + force_attraction
    return energy, force


r = np.arange(0.1, 5.0, 0.01)
lj_energy, lj_force = lj_potential(r, epsilon, sigma)


fig = plt.figure(figsize=(9, 6))
plt.plot(r, lj_energy, label='Lennard-Jones Energy')
plt.plot(r, lj_force, label='Lennard-Jones Force')
plt.xlabel("r (sigma)")
plt.ylabel("Lennard-Jones Energy & Force")
plt.legend()
plt.ylim(-3.0, 6.0)
plt.xlim(0.5, 3.0)
plt.savefig("ex3-potential-1-lj.png", dpi=300)
plt.show()
