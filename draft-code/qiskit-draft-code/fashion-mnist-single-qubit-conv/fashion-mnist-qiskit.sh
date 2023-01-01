#!/bin/bash --login
#SBATCH --job-name=testDaskJobQueue
#SBATCH --nodes=1
#SBATCH --time=03:30:00
#SBATCH --account=pawsey0419
#SBATCH --export=NONE
#SBATCH --mem=32G


module load hpc-python-collection/2022.11-py3.9.15

srun python fashion-mnist-classification-qiskit-prototype-ga.py