import math
import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA, COBYLA
import json
import time
import shutup
import pickle
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

from SPSAGradOptimiser.qiskit_opts.SPSA_Adam import ADAMSPSA
from qiskit.circuit import ParameterVector
import os

from token_mixing_layers import FourQubitParameterisedLayer
from two_by_two_local_tokens import LocalTokenMixing

def backboneCircFourQubitFeature(
        image_patches:Union[List[List[ParameterVector]], np.ndarray],
        single_patch_encoding_parameter: Union[ParameterVector, np.ndarray],
        single_patch_d_and_r_phase_parameter: Union[ParameterVector, np.ndarray],
        four_patch_d_and_r_phase_parameter: Union[ParameterVector, np.ndarray],
        two_patch_2_q_pqc_parameter: Union[ParameterVector, np.ndarray],
        finishing_layer_parameter:Union[ParameterVector, np.ndarray]
):
    """
    The backbone circuit for producing the four-qubit feature/representation of an 8 x 8 image.
    The first 4 qubits contain the feature, and the last 6 qubits are ancilla qubits, which, in an ideal simulation, should be reversed back to the |0> state.

    :param image_patches: All 16 image patches, each of which is a 4-element array (ParameterVector) reshaped into a 4 by 4 list, or a 4 by 4 by 4 ndarray, in which the last dimension is the original pixels.
    :param single_patch_encoding_parameter: "θ", 12N-element array, one-dim, where N is the number of "layers" of data-reuploading and L is the number of D&R repetitions in the single patch D&R
    :param single_patch_d_and_r_phase_parameter: "φ", 2-element array, one-dim, where L is the number of D&R repetitions in the single patch D&R
    :param four_patch_d_and_r_phase_parameter: 2-parameter array
    :param two_patch_2_q_pqc_parameter: "γ", 6N-element array, one-dim, where N is the number of layers of the two-patch 2-qubit parameterised layer
    :param finishing_layer_parameter: "η", 12K-element array, one-dim, where K is the number of layers of the finishing layer
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
    circ = QuantumCircuit(8, name="BackboneCirc4QFeature")
    local_mixing_count = 0
    for i in range(2):
        for j in range(2):
            local_patches = image_patches[i * 2:i * 2 + 2, j * 2:j * 2 + 2]
            circ.append(LocalTokenMixing(
                local_patches,
                single_patch_encoding_parameter,
                single_patch_d_and_r_phase_parameter,
                four_patch_d_and_r_phase_parameter,
                two_patch_2_q_pqc_parameter
            ), [local_mixing_count, local_mixing_count + 1, local_mixing_count + 2, local_mixing_count + 3, local_mixing_count + 4])
            local_mixing_count += 1

    circ.append(FourQubitParameterisedLayer(finishing_layer_parameter), [0, 1, 2, 3])

    return circ



if __name__ == '__main__':
    from two_by_two_local_tokens import TwoByTwoPatchLocalTokens
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

    num_single_patch_data_reuploading_layers = 2
    num_single_patch_d_and_r_repetitions = 2
    num_four_patch_d_and_r_repetitions = 2
    num_two_patch_2_q_pqc_layers = 2
    first_four_patch_pv = [[ParameterVector('x1',4),ParameterVector('x2',4)],[ParameterVector('x5',4),ParameterVector('x6',4)]]
    single_patch_encoding_parameter = ParameterVector('θ',
                                                      6 * num_single_patch_data_reuploading_layers * num_single_patch_d_and_r_repetitions)
    single_patch_d_and_r_phase_parameter = ParameterVector('φ', num_single_patch_d_and_r_repetitions)
    four_patch_d_and_r_phase_parameter = ParameterVector('ψ', num_four_patch_d_and_r_repetitions)
    two_patch_2_q_pqc_parameter = ParameterVector('γ', 6 * num_two_patch_2_q_pqc_layers)

    eta = ParameterVector('η', 12)


    backbone = backboneCircFourQubitFeature(
        patches,
        single_patch_encoding_parameter,
        single_patch_d_and_r_phase_parameter,
        four_patch_d_and_r_phase_parameter,
        two_patch_2_q_pqc_parameter,
        eta)
    backbone.draw(output='mpl', style='bw', filename='BackboneCirc4QFeature.png')



