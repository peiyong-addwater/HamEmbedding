import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from two_by_two_local_patches import TwoByTwoLocalPatches2, TwoByTwoLocalPatches
from four_qubit_token_mixing import FourQubitParameterisedLayer, PermutationInvariantFourQLayer, QFTTokenMixing

def LocalPatchesFixedDRWithQFTMixingDR(
        img_patches: Union[np.ndarray, jnp.ndarray, pnp.ndarray],
        encode_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        wires: Union[List[int], Wires]
):
    """
    Just the TwoByTwoLocalPatches2 function with QFT appended to the first four qubits
    The D&R process follows that in FourPixelDepositAndReverseFixed
    :param img_patches: four image patches, a 2 by 2 by 4 array
    :param encode_parameters: "θ", 6N*2*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2*2 element array
    :param local_patches_phase_parameters: "ψ", 2 element array
    :param wires: 7, first 1 have the deposited encoded information of the two by two local patches
    :return:
    """
    n_encode_params = len(encode_parameters)
    n_single_patch_phase_parameters = len(single_patch_phase_parameters)
    assert n_single_patch_phase_parameters == 4
    assert n_encode_params / 2 == n_encode_params // 2
    assert n_encode_params // 24 == n_encode_params / 24
    first_half_encode_params = encode_parameters[:n_encode_params // 2]
    second_half_encode_params = encode_parameters[n_encode_params // 2:]
    qml.Hadamard(wires[0])
    TwoByTwoLocalPatches2(img_patches, first_half_encode_params, single_patch_phase_parameters[:2],
                          wires=[wires[1], wires[2], wires[3], wires[4], wires[5], wires[6]])
    QFTTokenMixing(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.CPhase(local_patches_phase_parameters[0], wires=[wires[0], wires[1]])
    qml.adjoint(QFTTokenMixing)(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.adjoint(TwoByTwoLocalPatches2)(img_patches, first_half_encode_params, single_patch_phase_parameters[:2],
                          wires=[wires[1], wires[2], wires[3], wires[4], wires[5], wires[6]])
    qml.Barrier()
    qml.PauliX(wires[0])
    qml.Barrier()
    TwoByTwoLocalPatches2(img_patches, second_half_encode_params, single_patch_phase_parameters[2:],
                          wires=[wires[1], wires[2], wires[3], wires[4], wires[5], wires[6]])
    QFTTokenMixing(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.CPhase(local_patches_phase_parameters[1], wires=[wires[0], wires[1]])
    qml.adjoint(QFTTokenMixing)(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.adjoint(TwoByTwoLocalPatches2)(img_patches, second_half_encode_params, single_patch_phase_parameters[2:],
                          wires=[wires[1], wires[2], wires[3], wires[4], wires[5], wires[6]])



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.drawer.use_style("black_white")

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

    theta = np.random.randn(24)
    phi = np.random.randn(4)
    psi = np.random.randn(2)
    wires_local_mixing_dr = Wires(list(range(7)))
    dev1 = qml.device("default.qubit", wires = wires_local_mixing_dr)
    @qml.qnode(dev1)
    def circ():
        LocalPatchesFixedDRWithQFTMixingDR(
            first_four_patches,
            theta,
            phi,
            psi,
            wires_local_mixing_dr
        )
        return qml.state()


    print(qml.draw(circ)())


