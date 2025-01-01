#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#                                                                              #
##################### Author: Y-L WANG & Date: 2024-Feb-11 #####################

import numpy as np
import random
import math

numAtoms = 16
boxLength = 10

epsilon = 1.0  # energy in unit of k_BT
sigma = 1.0    # distance in unit of sigma

numSteps = 150
timeStep = 0.01

temperatureRef = 1.0

np.random.seed(numAtoms)
# positions = np.random.rand(numAtoms, 2)*boxLength # positions of all particles
# 16 particles are generated on regular 4x4 lattices
positions = np.zeros((numAtoms, 2))
numLatticesX = round(math.sqrt(numAtoms))
numLatticesY = math.ceil(numAtoms/numLatticesX)
for iatom in range(numAtoms):
	positions[iatom, 0] = (iatom%numLatticesX+0.5)*(boxLength/numLatticesX) + np.random.rand()*0.5
	positions[iatom, 1] = (math.floor(iatom/numLatticesX)+0.5)*(boxLength/numLatticesY) + np.random.rand()*0.5
velocities = np.random.rand(numAtoms, 2) - 0.5      # velocities of all particles

masses = np.ones(numAtoms)                          # masses of all particles
energies = np.zeros(numAtoms)                       # energies of all particles
forces = np.zeros((numAtoms, 2))                    # forces of all particles
accelerations = np.zeros((numAtoms, 2))             # accelerations of all particles


def update_positions(positions, velocities, timeStep):
	for iatom in range(numAtoms):
		for col in [0, 1]:
			positions[iatom, col] += timeStep*velocities[iatom, col]


boundaryCondition = 'periodic' # `periodic` or `reflective`
def boundary_conditions(positions, boxLength, boundaryCondition):
	for iatom in range(numAtoms):
		for col in [0, 1]:
			if boundaryCondition == 'periodic':
				if positions[iatom, col] > boxLength:
					positions[iatom, col] -= boxLength
				if positions[iatom, col] < 0:
					positions[iatom, col] += boxLength
			elif boundaryCondition == 'reflective':
				if positions[iatom, col] > boxLength:
					positions[iatom, col] -= 2.0*(positions[iatom, col] - boxLength)
					velocities[iatom, col] *= -1.0
				if positions[iatom, col] < 0:
					positions[iatom, col] += 2.0*(0.0 - positions[iatom, col])
					velocities[iatom, col] *= -1.0


def lj_potential(energies, forces, positions, epsilon, sigma):
	for iatom in range(numAtoms-1):
		for jatom in range(iatom+1, numAtoms):
			dist_square = 0.0 # square of distance
			for col in [0, 1]:
				dist_col = positions[iatom, col] - positions[jatom, col]
				if dist_col > boxLength/2.0:
					dist_col -= boxLength
				if dist_col < -boxLength/2.0:
					dist_col += boxLength
				dist_square += dist_col**2
			dist = math.sqrt(dist_square)

			energy = 4.0*epsilon*((sigma/dist)**12 - (sigma/dist)**6)
			energies[iatom] += 0.5*energy # Accumulate energy
			energies[jatom] += 0.5*energy 

			force_repulsion = 48.0*epsilon*(sigma/dist)**12
			force_attraction = -24.0*epsilon*(sigma/dist)**6
			force = force_repulsion + force_attraction

			for col in [0, 1]:
				dist_col = positions[iatom, col] - positions[jatom, col]
				if dist_col > boxLength/2.0:
					dist_col -= boxLength
				if dist_col < -boxLength/2.0:
					dist_col += boxLength
				forces[iatom, col] += (force*dist_col/dist_square)
				forces[jatom, col] -= (force*dist_col/dist_square)


def update_velocities(forces, masses, accelerations, velocities, timeStep):
	for iatom in range(numAtoms):
		for col in [0, 1]:
			accelerations[iatom, col] = forces[iatom, col]/masses[iatom]
			velocities[iatom, col] += accelerations[iatom, col]*timeStep


def thermostat(velocities, masses, temperatureRef):
	kineticEnergy = 0.0
	for iatom in range(numAtoms):
		ke_temp = 0.0
		for col in [0, 1]:
			ke_temp += velocities[iatom, col]**2
		ke_temp *= 0.5*masses[iatom]
		kineticEnergy += ke_temp
	temperatureTemp = kineticEnergy*2.0/(2.0*numAtoms)

	scaling_factor = math.sqrt(temperatureRef/temperatureTemp)
	for iatom in range(numAtoms):
		for col in [0, 1]:
			velocities[iatom, col] *= scaling_factor


############################ Code for Visualization ############################
 
from matplotlib import pyplot as plt
from celluloid import Camera

fig = plt.figure()
plt.xlim([0, boxLength])
plt.ylim([0, boxLength])
camera = Camera(fig)

colorsList = random.sample(range(1, numAtoms+1), k = len(positions) )
colors = colorsList

for istep in range(numSteps):
	plt.scatter(positions[:, 0], positions[:, 1], s=256, c=colors, cmap='plasma')
	camera.snap()
	update_positions(positions, velocities, timeStep)
	boundary_conditions(positions, boxLength, boundaryCondition)
	lj_potential(energies, forces, positions, epsilon, sigma)
	update_velocities(forces, masses, accelerations, velocities, timeStep)
	thermostat(velocities, masses, temperatureRef)

animation = camera.animate()
animation.save('ex3-potential-2-potential-thermostat.gif', writer = 'pillow', fps=64)

