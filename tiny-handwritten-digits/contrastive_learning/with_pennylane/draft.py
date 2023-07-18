import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt
from pennylane import classical_shadow, shadow_expval, ClassicalShadow

dev = qml.device("default.qubit", wires=2, shots=5000)

@qml.shadows.shadow_state(wires=[0, 1], diffable=True)
@qml.qnode(dev)
def circuit(x):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RX(x, wires=0)
    return qml.classical_shadow(wires=[0, 1])

x = np.array(1.2)
print(circuit(x))
print(qml.jacobian(lambda x: np.real(circuit(x)))(x))