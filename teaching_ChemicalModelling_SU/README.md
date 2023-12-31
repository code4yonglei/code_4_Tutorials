# Simple source codes for teaching

- [MMK_Gaussian]](#1)
- [MC simulation of LJ fluid](#2)
- [MD simulation of Ar fluid](#3)


<span id='1'></span>
## Gaussian Scripts

Scripts for running Gaussian calculations using the cluster.



<span id='2'></span>
## MC_LJ_Fluid (C & Gnuplot)

This simple code and the gnuplot script are used to simulate 1000 LJ particles using MC simulation technique.


---
***


<span id='3'></span>
## MD_Ar_Fluid (C++ & Python)

The source codes and scripts ares used to simulate 256 Ar atoms at liquid state in a simulation box and to perform statistical analysis of physical properties.

**Four** files are needed before perform the simulation.
- head.h
- main.cpp
- simu_plot.py
- simu_cord.input

**Two** files will be obtained from simulations.
- simu_ener.txt
- simu_traj.xyz

**Four** PNG figures will be avavlable after running the python script.
- simu_Energy.png
- simu_Momentum.png
- simu_Pressure.png
- simu_Temperature.png
