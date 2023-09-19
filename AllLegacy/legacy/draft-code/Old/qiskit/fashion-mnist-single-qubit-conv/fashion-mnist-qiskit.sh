#!/bin/bash --login
#SBATCH --job-name=qiskitFMNIST
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --account=pawsey0419
#SBATCH --export=NONE
#SBATCH --mem=64G


module load hpc-python-collection/2022.11-py3.9.15

srun python fashion-mnist-classification-qiskit-prototype-ga.py