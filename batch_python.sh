#!/bin/bash

#SBATCH --job-name=my_python_program
#SBATCH --nodes=1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16
#SBATCH --time=2880
#SBATCH --mail-type=ALL
#SBATCH --mail-user=uta-isabella.meyer@lip6.fr
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Source the modules initialization script
source /etc/profile.d/modules.sh

# Step 1: Load the Matlab module
module load python/anaconda3

python my_python_script.py
