set terminal win

set xlabel 'MC Steps'

plot 'Thermodynamics_MCsim.txt' u 1:2 w l lw 4 lc 1 dt 1 t "Averged energy per particle",\
     'Thermodynamics_MCsim.txt' u 1:3 w l lw 4 lc 2 dt 1 t "Pressure",\
     'Thermodynamics_MCsim.txt' u 1:4 w l lw 4 lc 4 dt 1 t "Number of accepted/Number of attempted"
