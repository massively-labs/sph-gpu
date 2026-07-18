set datafile separator comma
set terminal jpeg enhanced size 1200, 800 font "sans,16"
set output "workspace/plot/performance.jpeg"
set title "2D density evaluation: GPU and CPU baselines"
set xlabel "particles"
set ylabel "median time [ms]"
set logscale xy 2
set format x "2^{%L}"
set grid
set key top left

plot "workspace/dat/performance.csv" using 2:3 with linespoints linewidth 2 pointtype 7 \
       linecolor rgb "#2864dc" title "Massively GPU pipeline", \
     "workspace/dat/performance.csv" using 2:4 with linespoints linewidth 2 pointtype 5 \
       linecolor rgb "#2b9348" title "CPU cell list", \
     "workspace/dat/performance.csv" using 2:5 with linespoints linewidth 2 pointtype 9 \
       linecolor rgb "#dc3c32" title "CPU all pairs"
