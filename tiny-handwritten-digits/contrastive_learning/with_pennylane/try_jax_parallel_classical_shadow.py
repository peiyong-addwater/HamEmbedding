import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import jax
import numpy as np
from typing import List, Tuple, Union, Optional
from pennylane.wires import Wires
jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

dev = qml.device("default.qubit", wires=2, shots=10000)

@qml.shadows.shadow_state(wires=[0, 1], diffable=True)
@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(x, wires=0)
    return qml.classical_shadow(wires=[0, 1])

x = pnp.array(1.2)
print(circuit(x))
grad = qml.jacobian(lambda x: np.real(circuit(x)))(x)
print(grad)