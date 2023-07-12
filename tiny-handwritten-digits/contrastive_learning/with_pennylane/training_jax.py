import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import jax
import numpy as np
from typing import List, Tuple, Union, Optional
from pennylane.wires import Wires

import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from backbone_hierarchical_encoding_QFT_Mixing import backboneQFTMixing

def qnn(
        patched_img: Union[np.ndarray, jnp.ndarray, pnp.ndarray],
        encode_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        final_layer_parameters:Optional[Union[np.ndarray, jnp.ndarray, pnp.ndarray]] = None,
        final_layer_type: Optional[str] = None,
):
    pass

if __name__ == '__main__':
    #jax.config.update('jax_platform_name', 'cpu')
    #jax.config.update("jax_enable_x64", True)
    dev = qml.device('default.qubit', wires=2, shots=5)
    x = pnp.random.normal(loc=0, scale=1, size=(3, 1), requires_grad=True)
    y = pnp.random.normal(loc=0, scale=1, size=(3, 1), requires_grad=True)


    #@jax.jit
    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(x, y):
        qml.RX(x[0], wires=[0])
        qml.RY(x[1], wires=[0])
        qml.RX(y[0], wires=[1])
        qml.RY(y[1], wires=[1])
        qml.CNOT(wires=[0, 1])
        qml.RX(x[2], wires=[0])
        qml.RX(y[2], wires=[1])
        return qml.classical_shadow([0, 1])

    print(circuit(x, y))
