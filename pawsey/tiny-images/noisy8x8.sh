#!/bin/bash --login
#SBATCH --job-name=noisy8x8
#SBATCH --nodes=1
#SBATCH --time=3:00:00
#SBATCH --account=pawsey0419
#SBATCH --export=NONE
#SBATCH --mem=64G

module load hpc-python-collection/2022.11-py3.9.15

python noisy-conv-5x5-stride-3-image-8x8.py