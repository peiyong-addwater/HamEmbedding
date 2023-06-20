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

from two_by_two_patch_D_and_R import FourPixelDepositAndReverse

def TwoByTwoPatchLocalTokenMixing(
        img_patches:Union[List[List[ParameterVector]], np.ndarray],
        single_patch_encoding_parameter:Union[ParameterVector, np.ndarray],
        single_patch_d_and_r_phase_parameter:Union[ParameterVector, np.ndarray],
        to_gate = True
):
    """
    Encode 4 image patches into the phase of four qubits
    :param img_patches: 4 image patches, each of which is a 4-element array (ParameterVector) reshaped into a 2 by 2 list, or a 2 by 2 by 4 ndarray
    :param single_patch_encoding_parameter: "θ", 6N-element array, one-dim, where N is the number of "layers" of data-reuploading
    :param single_patch_d_and_r_phase_parameter: "φ", L-element array, one-dim, where L is the number of D&R repetitions
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(6, name = "TwoByTwoPatchLocalTokenMixing")
    assert len(img_patches) == 2
    assert len(img_patches[0]) == 2
    assert len(img_patches[1]) == 2
    patch_count = 0
    for i in range(2):
        for j in range(2):
            circ.append(FourPixelDepositAndReverse(img_patches[i][j], single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [patch_count, patch_count + 1, patch_count + 2])
            patch_count += 1

    return circ.to_gate(label="PatchLocalTokenMixing") if to_gate else circ

def FourQubitParameterisedLayer(parameters:Union[ParameterVector, np.ndarray]):
    """
    A four-qubit parameterised layer, which ends with circular entanglement with CZ gates
    :param parameters:
    :return:
    """
    pass


if __name__ == '__main__':
    def cut_8x8_to_2x2(img: np.ndarray):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattend patch
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

    first_four_patch_pv = [[ParameterVector('x1',4),ParameterVector('x2',4)],[ParameterVector('x5',4),ParameterVector('x6',4)]]
    theta = ParameterVector('θ', 12)
    phi = ParameterVector('φ', 2)
    local_token_mixing = TwoByTwoPatchLocalTokenMixing(first_four_patch_pv, theta, phi, to_gate=False)
    local_token_mixing.draw(output='mpl', style='bw', filename='TwoByTwoPatchLocalTokenMixing.png')

