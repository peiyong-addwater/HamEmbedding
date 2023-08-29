#!/bin/bash -l
#SBATCH --account=pawsey0419
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=230GB
#SBATCH --time=23:59:00

module --ignore-cache load gcc/12.1.0
module --ignore-cache load python/3.10.8 py-pip/22.2.2-py3.10.8 py-setuptools/57.4.0-py3.10.8

export PYTHONUSERBASE=/software/projects/pawsey0419/peiyongw/setonix/python
export PATH=$PATH:$PYTHONUSERBASE/bin

python3 train_pawsey_cpu.py --img_dir /scratch/pawsey0419/peiyongw/QML-ImageClassification/data/mini-digits/images --csv_file /scratch/pawsey0419/peiyongw/QML-ImageClassification/data/mini-digits/annotated_labels.csv --batch_size 50 --train_batches 20 --epochs 50 --n_mem_qubits 4 --n_patch_qubits 4 --L1 2 --L2 2 --L_MC 1 --reset_first_mem_qubit False --working_dir /scratch/pawsey0419/peiyongw/QML-ImageClassification/tiny-handwritten-digits/contrastive_learning