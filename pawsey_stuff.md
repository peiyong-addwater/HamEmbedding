# Pawsey Usage Quicknotes

## Python Modules

- To install packages, first load:
```shell
module load gcc/12.1.0
module load python/3.10.8 py-pip/22.2.2-py3.10.8 py-setuptools/57.4.0-py3.10.8
```

Then do
```shell
export PYTHONUSERBASE=/software/projects/pawsey0419/peiyongw/setonix/python
export PATH=$PATH:$PYTHONUSERBASE/bin
```
Then install packages using `pip install --user <package_name>`. We will need the following packages (for CPU usage):
```shell
pip install --user qiskit[machine-learning,visualization] qiskit-aer pennylane pennylane-qiskit pennylane-lightning
```

Install PyTorch for CPU usage:
```shell
pip install --user torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Other DL packages:
```shell
pip install --user lightly tensorboard
```

Don't forget to purge pip cache after installation (or during installation):
```shell
pip cache purge
```
## CUDA
### PyTorch
Maybe GPU on the login node?
```shell
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
### Qiskit
Command to install 'qiskit-aer-gpu':
```shell
module load gcc/12.1.0
module load python/3.10.8 py-pip/22.2.2-py3.10.8 py-setuptools/57.4.0-py3.10.8
module load cudatoolkit/22.3_11.6
pip install qiskit-aer-gpu
```

## Some Configurations

If running on Pawsey, to make relative imports work, we may need to add the following to Python files:
```python
import sys
sys.path.insert(0, '/scratch/pawsey0419/peiyongw/QML-ImageClassification')
```
Also need to configure the address to the data folder.



## Example Scripts

```shell
#!/bin/bash -l
#SBATCH --account=pawsey0419
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=230GB
#SBATCH --time=23:59:00

module load gcc/12.1.0
module load python/3.10.8 py-pip/22.2.2-py3.10.8 py-setuptools/57.4.0-py3.10.8

export PYTHONUSERBASE=/software/projects/pawsey0419/peiyongw/setonix/python
export PATH=$PATH:$PYTHONUSERBASE/bin

python3 train_pawsey_cpu.py 
```