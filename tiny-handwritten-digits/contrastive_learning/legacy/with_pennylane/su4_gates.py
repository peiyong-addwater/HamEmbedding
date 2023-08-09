import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import torch

def su4_gate(params:Union[np.ndarray, torch.Tensor, pnp.ndarray], wires:Union[Wires, List[int]]):
    """
    A 15-parameter, most generic 2-qubit gate
    :param params:
    :param wires:
    :return:
    """
    qml.U3(params[0], params[1], params[2], wires=wires[0])
    qml.U3(params[3], params[4], params[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[6], wires=wires[0])
    qml.RZ(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.U3(params[9], params[10], params[11], wires=wires[0])
    qml.U3(params[12], params[13], params[14], wires=wires[1])

def beheaded_su4_gate(params:Union[np.ndarray, torch.Tensor, pnp.ndarray], wires:Union[Wires, List[int]]):
    """

    :param params:
    :param wires:
    :return:
    """
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[0], wires=wires[0])
    qml.RZ(params[1], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(params[2], wires=wires[0])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.U3(params[3], params[4], params[5], wires=wires[0])
    qml.U3(params[6], params[7], params[8], wires=wires[1])

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    qml.drawer.use_style("black_white")

    # draw the circuits
    dev = qml.device("default.qubit", wires=2)
    wires = Wires([0, 1])
    @qml.qnode(dev)
    def circuit_su4(params):
        su4_gate(params, wires=wires)
        return qml.state()

    su4_params = np.random.randn(15)
    fig, ax = qml.draw_mpl(circuit_su4)(su4_params)
    plt.savefig("su4_circuit.png")
    plt.close(fig)

    bsu4_params = np.random.randn(9)
    @qml.qnode(dev)
    def circuit_bsu4(params):
        beheaded_su4_gate(params, wires=wires)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit_bsu4)(bsu4_params)
    plt.savefig("beheaded_su4_circuit.png")
    plt.close(fig)




