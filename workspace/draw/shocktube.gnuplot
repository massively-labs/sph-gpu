set datafile separator comma
set terminal jpeg enhanced size 1200, 800 font "sans,16"
set output "workspace/plot/shocktube.jpeg"
set title "Sod shock tube: compressible SPH on Massively"
set xlabel "x"
set ylabel "density"
set xrange [0:1]
set yrange [0:1.1]
set grid
set key top right

plot "workspace/dat/shocktube.csv" using 1:2 with points pointtype 7 pointsize 0.45 \
       linecolor rgb "#2864dc" title "SPH", \
     "workspace/dat/shocktube.csv" using 1:6 with lines linewidth 2.5 \
       linecolor rgb "#dc3c32" title "exact Sod"
