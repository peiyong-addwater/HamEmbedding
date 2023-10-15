import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
from typing import List, Tuple, Union, Optional
from pennylane.wires import Wires
import torch
from utils import Reset0
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from two_by_two_local_patches import TwoByTwoLocalPatches2, TwoByTwoLocalPatches
from four_qubit_token_mixing import FourQubitParameterisedLayer, PermutationInvariantFourQLayer, QFTTokenMixing

def LocalPatchesFixedDRWithQFTMixingDR(
        img_patches: Union[np.ndarray,  pnp.ndarray, torch.Tensor],
        encode_parameters: Union[np.ndarray, pnp.ndarray,  torch.Tensor],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray,  torch.Tensor],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray,  torch.Tensor],
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
    Reset0(wires=[wires[1], wires[2], wires[3], wires[4]])

    qml.Barrier()
    qml.PauliX(wires[0])
    qml.Barrier()
    TwoByTwoLocalPatches2(img_patches, second_half_encode_params, single_patch_phase_parameters[2:],
                          wires=[wires[1], wires[2], wires[3], wires[4], wires[5], wires[6]])
    QFTTokenMixing(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.CPhase(local_patches_phase_parameters[1], wires=[wires[0], wires[1]])
    Reset0(wires=[wires[1], wires[2], wires[3], wires[4]])
    qml.Barrier()

def backboneQFTMixing(
        patched_img: Union[np.ndarray, torch.Tensor, pnp.ndarray],
        encode_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray, torch.Tensor],
        final_layer_parameters:Optional[Union[np.ndarray, torch.Tensor, pnp.ndarray]] = None,
        final_layer_type: Optional[str] = None,
        wires: Union[List[int], Wires] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
):
    """
    The backbone of the hierarchical encoding with QFT mixing for each 4 adjacent patches.
    If final_layer_parameters is not none, then an additional layer will be added at the end of the first 4 qubits,
    after another QFT mixing.
    The first 4 qubits contain the encoded information of the image,
    and the last 6 qubits are ancilla qubits.
    :param final_layer_type:
    :param patched_img: All 16 image patches, a 4 by 4 by 4 array
    :param encode_parameters: "θ", 6N*2*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2*2 element array
    :param local_patches_phase_parameters: "ψ", 2 element array
    :param final_layer_parameters: "η" 12L parameters if using FourQubitParameterisedLayer, 3L parameters if using PermutationInvariantFourQLayer
    :param wires: 10 wires, first 4 have the deposited encoded information of the image
    :return:
    """
    """
        For an image:
            [[ 0  1  2  3  4  5  6  7]
            [ 8  9 10 11 12 13 14 15]
            [16 17 18 19 20 21 22 23]
            [24 25 26 27 28 29 30 31]
            [32 33 34 35 36 37 38 39]
            [40 41 42 43 44 45 46 47]
            [48 49 50 51 52 53 54 55]
            [56 57 58 59 60 61 62 63]]
        the patches (4 by 4 by 4 array):
            [[[ 0.  1.  8.  9.]
            [ 2.  3. 10. 11.]
            [ 4.  5. 12. 13.]
            [ 6.  7. 14. 15.]]

            [[16. 17. 24. 25.]
            [18. 19. 26. 27.]
            [20. 21. 28. 29.]
            [22. 23. 30. 31.]]

            [[32. 33. 40. 41.]
            [34. 35. 42. 43.]
            [36. 37. 44. 45.]
            [38. 39. 46. 47.]]

            [[48. 49. 56. 57.]
            [50. 51. 58. 59.]
            [52. 53. 60. 61.]
            [54. 55. 62. 63.]]]
    """
    n_encode_params = len(encode_parameters)
    n_single_patch_phase_parameters = len(single_patch_phase_parameters)
    assert n_single_patch_phase_parameters == 4
    assert n_encode_params / 2 == n_encode_params // 2
    assert n_encode_params // 24 == n_encode_params / 24
    local_mixing_count = 0
    for i in range(2):
        for j in range(2):
            local_patches = patched_img[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            LocalPatchesFixedDRWithQFTMixingDR(local_patches,
                                               encode_parameters,
                                               single_patch_phase_parameters,
                                               local_patches_phase_parameters,
                                               wires=[wires[local_mixing_count], wires[local_mixing_count+1],wires[local_mixing_count+2],wires[local_mixing_count+3], wires[local_mixing_count+4], wires[local_mixing_count+5], wires[local_mixing_count+6]]
                                               )
            qml.Barrier()
            local_mixing_count += 1
    if final_layer_parameters is not None:
        if final_layer_type == "generic":
            QFTTokenMixing(wires=[wires[0], wires[1], wires[2], wires[3]])
            FourQubitParameterisedLayer(final_layer_parameters, wires=[wires[0], wires[1], wires[2], wires[3]])
        elif final_layer_type == "permutation-invariant":
            PermutationInvariantFourQLayer(final_layer_parameters, wires=[wires[0], wires[1], wires[2], wires[3]])
        else:
            raise ValueError("final_layer_type must be either 'generic' or 'permutation-invariant'")
    else:
        QFTTokenMixing(wires=[wires[0], wires[1], wires[2], wires[3]])





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
    eta = np.random.rand(12)
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
    fig, ax = qml.draw_mpl(circ)()
    plt.savefig("LocalPatchesFixedDRWithQFTMixingDR.png")
    plt.close(fig)

    wires_backbone = Wires(list(range(10)))
    dev2 = qml.device('default.qubit', wires = wires_backbone)
    @qml.qnode(dev2)
    def circ_backbone():
        backboneQFTMixing(
            patches,
            theta,
            phi,
            psi,
            eta,
            final_layer_type="generic"
        )
        return qml.state()


    print(qml.draw(circ_backbone)())
    fig, ax = qml.draw_mpl(circ_backbone)()
    plt.savefig("backbone.png")
    plt.close(fig)


