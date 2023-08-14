import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

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
    A 9-parameter, 2-qubit gate
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

def FourQubitGenericParameterisedLayer(
    parameters: torch.Tensor,
    wires:Union[Wires, List[int]]
):
    """
    a u3 layer followed by three beheaded_su4_gate layers
    total number of parameters: 3*4 + 3*9 = 39
    :param parameters:
    :param wires:
    :return:
    """
    single_qubit_params = parameters[:3*4] # u3 layer
    beheaded_su4_param_1 = parameters[3*4:3*4+9]
    beheaded_su4_param_2 = parameters[3*4+9:3*4+9+9]
    beheaded_su4_param_3 = parameters[3*4+9+9:3*4+9+9+9]

    qml.U3(single_qubit_params[0], single_qubit_params[1], single_qubit_params[2], wires=wires[0])
    qml.U3(single_qubit_params[3], single_qubit_params[4], single_qubit_params[5], wires=wires[1])
    qml.U3(single_qubit_params[6], single_qubit_params[7], single_qubit_params[8], wires=wires[2])
    qml.U3(single_qubit_params[9], single_qubit_params[10], single_qubit_params[11], wires=wires[3])

    beheaded_su4_gate(beheaded_su4_param_1, wires=[wires[0], wires[1]])
    beheaded_su4_gate(beheaded_su4_param_2, wires=[wires[1], wires[2]])
    beheaded_su4_gate(beheaded_su4_param_3, wires=[wires[2], wires[3]])

def BeheadedSU4Chain(
        parameters: torch.Tensor,
        wires:Union[Wires, List[int]]
):
    """
    A chain of three beheaded_su4_gate layers
    :param parameters:
    :param wires:
    :return:
    """
    n_layers = len(wires)-1
    for i in range(n_layers):
        beheaded_su4_gate(parameters[9*i:9*(i+1)], wires=[wires[i], wires[i+1]])

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    dev = qml.device("default.qubit", wires=4)

    su4_param = torch.randn(15)
    beheaded_su4_param = torch.randn(9)
    four_qubit_param = torch.randn(39)
    bsu4_param_chain = torch.randn(27)

    @qml.qnode(dev, interface="torch")
    def all_param_layer_circuits():
        su4_gate(su4_param, wires=[0, 1])
        qml.Barrier()
        beheaded_su4_gate(beheaded_su4_param, wires=[1,2])
        qml.Barrier()
        FourQubitGenericParameterisedLayer(four_qubit_param, wires=[0, 1, 2, 3])
        qml.Barrier()
        BeheadedSU4Chain(bsu4_param_chain, wires=[0, 1, 2, 3])
        return qml.state()


    fig, ax = qml.draw_mpl(all_param_layer_circuits)()
    plt.savefig("two_q_and_four_q_pqc.png")
    plt.close(fig)