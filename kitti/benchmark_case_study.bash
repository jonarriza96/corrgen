#!/bin/bash

# List of n_corrgen values
n_corrgen_values=(3 6 8 9 12 24)

# Loop through each n_corrgen value
for n_corrgen_value in "${n_corrgen_values[@]}"
do
    # Run the Python command with the current n_corrgen value
    python kitti.py --save --case 2  --no_visualization --lp --n_corrgen "$n_corrgen_value"   
done

