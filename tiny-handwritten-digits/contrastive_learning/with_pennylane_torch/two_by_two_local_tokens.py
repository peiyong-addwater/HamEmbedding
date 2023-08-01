import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
from reset_gate import Reset0
from token_mixing_layers import FourQubitGenericParameterisedLayer, su4_gate
from two_by_two_patch_d_and_r import FourPixelOneQubit
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def FourPatchOneQ(
        patch1: torch.Tensor,
        patch2: torch.Tensor,
        patch3: torch.Tensor,
        patch4: torch.Tensor,
        single_patch_encoding_parameter: torch.Tensor,
        single_patch_d_and_r_phase_parameter: torch.Tensor,
        phase_parameter: torch.Tensor,
        two_patch_2_q_pqc_parameter: torch.Tensor,
        wires: Union[List[int], Wires]
):
    """
    Encode four image patches in to one qubit with deposit and reset (two reps, one rep for two patches)
    :param patch1: 4-element array
    :param patch2: 4-element array
    :param patch3: 4-element array
    :param patch4: 4-element array
    :param single_patch_encoding_parameter: 6N*2 element array, where N is the number of data re-uploading repetitions
    :param single_patch_d_and_r_phase_parameter: 2 element array
    :param phase_parameter: 2 element array
    :param two_patch_2_q_pqc_parameter: 15-parameter array, since we are using su4 gate
    :param wires: 5 qubits in total
    :return:
    """
    qml.Hadamard(wires[0])
    FourPixelOneQubit(patch1, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, wires=[wires[1], wires[2], wires[3]])
    qml.Barrier()
    FourPixelOneQubit(patch2, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, wires=[wires[2], wires[3], wires[4]])
    qml.Barrier()
    su4_gate(two_patch_2_q_pqc_parameter, wires=[wires[1], wires[2]])
    qml.Barrier()
    qml.CPhase(phase_parameter[0], wires=[wires[0], wires[1]])
    qml.Barrier()
    Reset0(wires=[wires[1], wires[2]]) # reset the two patch qubits, wires 3 and 4 are already reset by the FourPixelOneQubit circuit
    qml.Barrier()
    qml.PauliX(wires=wires[0])
    qml.Barrier()
    FourPixelOneQubit(patch3, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, wires=[wires[1], wires[2], wires[3]])
    qml.Barrier()
    FourPixelOneQubit(patch4, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, wires=[wires[2], wires[3], wires[4]])
    qml.Barrier()
    su4_gate(two_patch_2_q_pqc_parameter, wires=[wires[1], wires[2]])
    qml.Barrier()
    qml.CPhase(phase_parameter[1], wires=[wires[0], wires[1]])
    qml.Barrier()
    Reset0(wires=[wires[1], wires[2]]) # reset the two patch qubits, wires 3 and 4 are already reset by the FourPixelOneQubit circuit


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    x = torch.randn(2,2,4)
    single_patch_encoding_parameter = torch.randn(12)
    single_patch_d_and_r_phase_parameter = torch.randn(2)
    phase_parameter = torch.randn(2)
    two_patch_2_q_pqc_parameter = torch.randn(15)
    wires = Wires([0, 1, 2, 3, 4])
    dev = qml.device("default.qubit", wires=wires)
    @qml.qnode(dev, interface="torch")
    def circ_draw():
        FourPatchOneQ(x[0,0], x[0,1], x[1,0], x[1,1], single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, phase_parameter, two_patch_2_q_pqc_parameter, wires=wires)
        return qml.state()


    fig, ax = qml.draw_mpl(circ_draw)()
    plt.savefig("four_patches_one_qubit.png")
    plt.close(fig)