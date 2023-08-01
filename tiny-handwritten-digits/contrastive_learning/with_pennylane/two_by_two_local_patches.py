import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import torch
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from two_by_two_patch_D_and_R import FourPixelDepositAndReset, FourPixelDepositAndResetFixed

def TwoByTwoLocalPatches(
        img_patches: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        single_patch_encode_parameters: Union[torch.Tensor, np.ndarray, pnp.ndarray],
        single_patch_d_and_r_parameter: Union[torch.Tensor, np.ndarray, pnp.ndarray],
        wires: Union[List[int], Wires]
):
    """
    Encode 4 adjacent patches in to the phase of four qubits
    :param img_patches: four image patches, a 2 by 2 by 4 array
    :param single_patch_encode_parameters: "θ", 6NL-element array, one-dim, where N is the number of "layers" of data-reuploading and L is the number of D&R repetitions in the single patch D&R
    :param single_patch_d_and_r_parameter: "φ", L-element array, one-dim, where L is the number of D&R repetitions
    :param wires: 6 qubits, the first four will have the four local patches encoded as phases on the qubit
    :return:
    """
    assert len(img_patches) == 2
    assert len(img_patches[0]) == 2
    assert len(img_patches[1]) == 2
    patch_count = 0
    for i in range(2):
        for j in range(2):
            FourPixelDepositAndReset(img_patches[i][j], single_patch_encode_parameters,
                                     single_patch_d_and_r_parameter,
                                     wires=[wires[patch_count], wires[patch_count + 1], wires[patch_count + 2]])
            qml.Barrier()
            patch_count += 1

def TwoByTwoLocalPatches2(
        img_patches: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        encode_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        wires: Union[List[int], Wires]
):
    """
    Use the FourPixelDepositAndReverseFixed function to encode four (two by two) local potches into four single-qubit states
    :param img_patches: four image patches, a 2 by 2 by 4 array
    :param encode_parameters: "θ", 6N*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2 element array
    :param wires: 6
    :return:
    """
    assert len(img_patches) == 2
    assert len(img_patches[0]) == 2
    assert len(img_patches[1]) == 2
    patch_count = 0
    for i in range(2):
        for j in range(2):
            FourPixelDepositAndResetFixed(img_patches[i][j], encode_parameters, single_patch_phase_parameters,
                                          wires=[wires[patch_count], wires[patch_count + 1], wires[patch_count + 2]])

            qml.Barrier()
            patch_count += 1


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

    wires = Wires([0, 1, 2, 3, 4, 5])
    dev = qml.device("default.qubit", wires=wires)


    def cut_8x8_to_2x2(img: np.ndarray):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattened patch
        patches = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                patches[i, j] = img[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
        return patches


    img = np.arange(64).reshape(8, 8)
    patches = cut_8x8_to_2x2(img)
    print(patches)
    print(img)
    first_four_patches = patches[:2, :2]
    print(first_four_patches)
    for i in range(2):
        for j in range(2):
            print("Patch ", i, j)
            print(patches[i * 2:i * 2 + 2, j * 2:j * 2 + 2])

    theta = np.random.randn(6)
    phi = np.random.randn(1)

    @qml.qnode(dev)
    def circuit(theta, phi):
        TwoByTwoLocalPatches(first_four_patches, theta, phi, wires)
        return qml.state()

    fig, ax = qml.draw_mpl(circuit)(theta, phi)
    plt.savefig("two_by_two_local_patches_encoded.png")
    plt.close(fig)

    theta = np.random.randn(12)
    phi = np.random.randn(2)


    @qml.qnode(dev)
    def circuit(theta, phi):
        TwoByTwoLocalPatches2(first_four_patches, theta, phi, wires)
        return qml.state()


    fig, ax = qml.draw_mpl(circuit)(theta, phi)
    plt.savefig("two_by_two_local_patches_encoded_fixed_d_and_r.png")
    plt.close(fig)
