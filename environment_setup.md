# Setting Up the Environment

## Create a virtual environment

```bash
conda create -n QMLGPU python=3.11
conda activate QMLGPU
```
## Install the required packages

### Tensorflow GPU

Following https://www.tensorflow.org/install/pip

```bash
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# Verify install:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Jax GPU

Following https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier

```bash
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
### PyTorch GPU

Following https://pytorch.org/get-started/locally/

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Qiskit with GPU

```bash
pip install qiskit[visualization]
pip install qiskit-ibm-provider qiskit-ibm-runtime
pip install qiskit-machine-learning qiskit-algorithms qiskit-dynamics
conda install -c conda-forge custatevec
```

### TensorCircuit

```bash
pip install tensorcircuit cotengra
```

### Jax NN Library

```bash
pip install optax flax dm-haiku equinox dm-pix augmax
```

### PyTorch SSL

```bash
pip install lightning lightly
```

### Pennylane

Be sure to install Pennylane at the end, since there is a restriction on NumPy version for Pennylane v0.32.

```bash
pip install pennylane --upgrade
pip install pennylane-lightning pennylane-lightning[gpu] pennylane-qiskit
```