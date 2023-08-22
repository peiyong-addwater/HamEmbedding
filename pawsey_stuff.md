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

## Some Configurations

If running on Pawsey, to make relative imports work, we may need to add the following to Python files:
```python
import sys
sys.path.insert(0, '/scratch/pawsey0419/peiyongw/QML-ImageClassification')
```
Also need to configure the address to the data folder.