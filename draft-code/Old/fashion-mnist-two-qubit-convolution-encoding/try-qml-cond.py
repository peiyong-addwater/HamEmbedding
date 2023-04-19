import pennylane as qml
from pennylane import numpy as np

dev = qml.device("default.qubit", wires=3)

@qml.qnode(dev)
def qnode(x, y):
    qml.Hadamard(0)
    m_0 = qml.measure(0)
    qml.cond(m_0, qml.RY)(x, wires=1)

    qml.Hadamard(2)
    qml.RY(-np.pi/2, wires=[2])
    m_1 = qml.measure(2)
    qml.cond((m_1 == 0)&(m_0==1), qml.RX)(y, wires=1)
    return qml.expval(qml.PauliZ(1))

first_par = np.array(0.3, requires_grad=True)
sec_par = np.array(1.23, requires_grad=True)
print(qnode(first_par, sec_par))