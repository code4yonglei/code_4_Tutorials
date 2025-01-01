#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#                                                                              #
##################### Author: Y-L WANG & Date: 2024-Feb-11 #####################

import numpy as np
import random

numAtoms = 16
boxLength = 10

np.random.seed(numAtoms)
positions = np.random.rand(numAtoms, 2)*boxLength # positions of all particles

############################ Code for Visualization ############################
 
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
plt.xlim([0, boxLength])
plt.ylim([0, boxLength])
camera = Camera(fig)

colorsList = random.sample(range(1, numAtoms+1), k = len(positions) )
colors = colorsList

plt.scatter(positions[:, 0], positions[:, 1], s=256, c=colors, cmap='plasma')
camera.snap()

animation = camera.animate()
animation.save('ex1-initialization.gif', writer = 'pillow', fps=64)
