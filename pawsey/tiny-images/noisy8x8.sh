#!/bin/bash --login
#SBATCH --job-name=noisy8x8
#SBATCH --partition=highmem
#SBATCH --nodes=2
#SBATCH --time=24:00:00
#SBATCH --account=pawsey0419
#SBATCH --export=NONE
#SBATCH --mem=1024G

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_PLACES=cores     #To bind threads to cores
export OMP_PROC_BIND=close  #To bind (fix) threads (allocating them as close as possible). This option works together with the "places" indicated above, then: allocates threads in closest cores.


module load hpc-python-collection/2022.11-py3.9.15

echo $OMP_NUM_THREADS

srun -N 2 -n 1 python noisy-conv-5x5-stride-3-image-8x8.py