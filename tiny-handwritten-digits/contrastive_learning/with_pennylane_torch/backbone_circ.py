import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
from reset_gate import Reset0
from token_mixing_layers import FourQubitGenericParameterisedLayer, su4_gate
from two_by_two_local_tokens import FourPatchOneQ
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

def SSLCircFourQubitZ(
        image_patches:torch.Tensor,
        single_patch_encoding_parameter: torch.Tensor,
        single_patch_d_and_r_phase_parameter: torch.Tensor,
        phase_parameter: torch.Tensor,
        two_patch_2_q_pqc_parameter: torch.Tensor,
        projection_head_parameter: torch.Tensor,
        wires: Union[List[int], Wires]
):
    """
    This SSL circuit produce a 4-qubit z.
    :param image_patches: a 4 by 4 by 4 tensor of image patches, where the last index is the original pixels
    :param single_patch_encoding_parameter: 6N*2 element array, where N is the number of data re-uploading repetitions
    :param single_patch_d_and_r_phase_parameter: 2 element array
    :param phase_parameter: 2 element array
    :param two_patch_2_q_pqc_parameter: 15-parameter array, since we are using su4 gate
    :param projection_head_parameter: using the FourQubitGenericParameterisedLayer, 39-parameter array
    :param wires: 8 qubits in total, first 4 qubits are for the z representation as in the SimCLR paper
    :return:
    """
    local_mixing_count = 0
    for i in range(2):
        for j in range(2):
            local_patches = image_patches[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            FourPatchOneQ(
                local_patches[0, 0],
                local_patches[0, 1],
                local_patches[1, 0],
                local_patches[1, 1],
                single_patch_encoding_parameter,
                single_patch_d_and_r_phase_parameter,
                phase_parameter,
                two_patch_2_q_pqc_parameter,
                wires=[wires[local_mixing_count], wires[local_mixing_count+1], wires[local_mixing_count+2], wires[local_mixing_count+3], wires[local_mixing_count+4]]
            )
            local_mixing_count += 1
    qml.Barrier()
    # token mixing with QFT, state after QFT is the h representation in the SimCLR paper
    qml.QFT(wires=wires[0:4])
    qml.Barrier()
    # projection head
    FourQubitGenericParameterisedLayer(projection_head_parameter, wires=wires[0:4])

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    def cut_8x8_to_2x2(img: torch.Tensor):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattened patch
        patches = torch.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                patches[i, j] = img[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
        return patches


    img = torch.arange(64).reshape(8, 8)
    patches = cut_8x8_to_2x2(img)
    print(patches)
    print(img)

    single_patch_encoding_parameter = torch.randn((6 * 2) * 1)
    single_patch_d_and_r_phase_parameter = torch.randn((2) * 1)
    phase_parameter = torch.randn((2) * 1)
    two_patch_2_q_pqc_parameter = torch.randn((15) * 1)
    projection_head_parameter = torch.randn((39) * 1)

    wires = [i for i in range(8)]
    dev = qml.device('default.mixed', wires=wires)
    @qml.qnode(dev, interface='torch')
    def circuit():
        SSLCircFourQubitZ(
            patches,
            single_patch_encoding_parameter,
            single_patch_d_and_r_phase_parameter,
            phase_parameter,
            two_patch_2_q_pqc_parameter,
            projection_head_parameter,
            wires
        )
        return qml.probs(wires=wires[0:4])

    print(circuit())
    original_stdout = sys.stdout
    with open('ssl_circ.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        print(qml.draw(circuit,decimals=None,max_length=9999999)())
        sys.stdout = original_stdout

