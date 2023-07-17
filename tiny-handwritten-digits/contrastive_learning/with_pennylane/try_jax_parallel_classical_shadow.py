import pennylane as qml
import pennylane.numpy as np
from matplotlib import pyplot as plt
from pennylane import classical_shadow, shadow_expval, ClassicalShadow

np.random.seed(666)

H = qml.Hamiltonian([1., 1.], [qml.PauliZ(0) @ qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

dev = qml.device("default.qubit", wires=range(2), shots=10000)
@qml.qnode(dev, interface="autograd")
def qnode(x, H):
    qml.Hadamard(0)
    qml.CNOT((0,1))
    qml.RX(x, wires=0)
    return shadow_expval(H)

x = np.array(0.5, requires_grad=True)

print(qnode(x, H), qml.grad(qnode)(x, H))

Hs = [H, qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)]
print(qnode(x, Hs))
print(qml.jacobian(qnode)(x, Hs))

