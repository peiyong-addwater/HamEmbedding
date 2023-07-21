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

from two_by_two_patch_D_and_R import FourPixelDepositAndReset
from token_mixing_layers import FourQubitParameterisedLayer, TwoQubitParameterisedLayer

def FourPatchOneQ(
        img_patch1:Union[List, np.ndarray, ParameterVector],
        img_patch2:Union[List, np.ndarray, ParameterVector],
        img_patch3:Union[List, np.ndarray, ParameterVector],
        img_patch4:Union[List, np.ndarray, ParameterVector],
        single_patch_encoding_parameter:Union[ParameterVector, np.ndarray],
        single_patch_d_and_r_phase_parameter:Union[ParameterVector, np.ndarray],
        four_patch_d_and_r_phase_parameter:Union[ParameterVector, np.ndarray],
        two_patch_2_q_pqc_parameter:Union[ParameterVector, np.ndarray],
        to_gate = True
):
    """
    Encode four image patches in to one qubit with deposit and reset.
    Each patch is encode into one qubit with the FourPixelDepositAndReset circuit.
    Then, the first two patches are mixed with the TwoQubitParameterisedLayer circuit, and encode into the phase
    of the |0> state of the deposit qubit and the last two patches are mixed with the TwoQubitParameterisedLayer circuit,
    and encode into the phase of the |1> state of the deposit qubit.
    Finally, the deposit qubit holds the information of all four patches.
    number of qubits required: 1 (for deposit) + (1+2+1)
    :param img_patch1:
    :param img_patch2:
    :param img_patch4:
    :param img_patch3:
    :param single_patch_encoding_parameter: 12N parameter array, where N is the number of layers of the single patch data re-uploading
    :param single_patch_d_and_r_phase_parameter: 2-parameter array
    :param four_patch_d_and_r_phase_parameter: 2-parameter array
    :param two_patch_2_q_pqc_parameter: 6N-parameter array, where N is the number of layers of the two-patch 2-qubit parameterised layer
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(5, name = "FourPatchOneQ")
    circ.h(0)
    circ.append(FourPixelDepositAndReset(img_patch1, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [1, 2, 3])
    circ.append(FourPixelDepositAndReset(img_patch2, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [2, 3, 4])
    circ.append(TwoQubitParameterisedLayer(two_patch_2_q_pqc_parameter, to_gate=to_gate), [1, 2])
    circ.cp(four_patch_d_and_r_phase_parameter[0], 0, 1)
    circ.reset([1, 2]) # qubits 3 and 4 are already reset by the FourPixelDepositAndReset circuit
    circ.x(0)
    circ.append(FourPixelDepositAndReset(img_patch3, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [1, 2, 3])
    circ.append(FourPixelDepositAndReset(img_patch4, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [2, 3, 4])
    circ.append(TwoQubitParameterisedLayer(two_patch_2_q_pqc_parameter, to_gate=to_gate), [1, 2])
    circ.cp(four_patch_d_and_r_phase_parameter[1], 0, 1)
    circ.reset([1, 2])  # qubits 3 and 4 are already reset by the FourPixelDepositAndReset circuit
    return circ.to_instruction(label="FourPatchOneQ") if to_gate else circ


def TwoByTwoPatchLocalTokens(
        img_patches:Union[List[List[ParameterVector]], np.ndarray],
        single_patch_encoding_parameter:Union[ParameterVector, np.ndarray],
        single_patch_d_and_r_phase_parameter:Union[ParameterVector, np.ndarray],
        to_gate = True
):
    """
    Encode 4 image patches into the phase of four qubits
    :param img_patches: 4 image patches, each of which is a 4-element array (ParameterVector) reshaped into a 2 by 2 list, or a 2 by 2 by 4 ndarray
    :param single_patch_encoding_parameter: "θ", 6NL-element array, one-dim, where N is the number of "layers" of data-reuploading and L is the number of D&R repetitions in the single patch D&R
    :param single_patch_d_and_r_phase_parameter: "φ", L-element array, one-dim, where L is the number of D&R repetitions
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(6, name = "TwoByTwoPatchLocalTokens")
    assert len(img_patches) == 2
    assert len(img_patches[0]) == 2
    assert len(img_patches[1]) == 2
    patch_count = 0
    for i in range(2):
        for j in range(2):
            circ.append(FourPixelDepositAndReset(img_patches[i][j], single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate), [patch_count, patch_count + 1, patch_count + 2])
            patch_count += 1

    return circ.to_instruction(label="TwoByTwoPatchLocalTokens") if to_gate else circ

def LocalTokenMixing(
        img_patches:Union[List[List[ParameterVector]], np.ndarray],
        single_patch_encoding_parameter:Union[ParameterVector, np.ndarray],
        single_patch_d_and_r_phase_parameter:Union[ParameterVector, np.ndarray],
        four_q_param_layer_parameter:Union[ParameterVector, np.ndarray],
        local_token_mixing_phase_parameter:Union[ParameterVector, np.ndarray, float],
        to_gate = True
):
    """
    Mix the local tokens in the four qubits at the end of the TwoByTwoPatchLocalTokens circuit with a parameterised layer
    :param img_patches:  4 image patches, each of which is a 4-element array (ParameterVector) reshaped into a 2 by 2 list, or a 2 by 2 by 4 ndarray
    :param single_patch_encoding_parameter: "θ", 6NL-element array, one-dim, where N is the number of "layers" of data-reuploading and L is the number of D&R repetitions in the single patch D&R
    :param single_patch_d_and_r_phase_parameter: "φ", L-element array, one-dim, where L is the number of D&R repetitions in the single patch D&R
    :param four_q_param_layer_parameter: "γ", 12K-element array, one-dim, where K is the number of layers of the 4-q parameterised layer
    :param local_token_mixing_phase_parameter: "ω", a one-element array for the phase parameter of the token mixing D&R
    :param to_gate:
    :return:
    """
    circ = QuantumCircuit(7, name="LocalTokenMixing")
    circ.h(0)
    local_tokens = TwoByTwoPatchLocalTokens(img_patches, single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, to_gate=to_gate)
    four_q_layer = FourQubitParameterisedLayer(four_q_param_layer_parameter, to_gate=to_gate)
    circ.append(local_tokens, [1, 2, 3, 4, 5, 6])
    circ.append(four_q_layer, [1, 2, 3, 4])
    circ.cp(local_token_mixing_phase_parameter[0], 0, 1)
    circ.reset([1, 2, 3, 4, 5, 6])

    return circ.to_instruction(label="LocalTokenMixing") if to_gate else circ

if __name__ == '__main__':
    def cut_8x8_to_2x2(img: np.ndarray):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattened patch
        patches = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                patches[i, j] = img[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
        return patches


    first_four_patch_pv = [ParameterVector('x1',4),ParameterVector('x2',4),ParameterVector('x5',4),ParameterVector('x6',4)]

    num_single_patch_data_reuploading_layers = 2
    num_single_patch_d_and_r_repetitions = 2
    num_four_patch_d_and_r_repetitions = 2
    num_two_patch_2_q_pqc_layers = 2

    single_patch_encoding_parameter = ParameterVector('θ', 6 * num_single_patch_data_reuploading_layers * num_single_patch_d_and_r_repetitions)
    single_patch_d_and_r_phase_parameter = ParameterVector('φ', num_single_patch_d_and_r_repetitions)
    four_patch_d_and_r_phase_parameter = ParameterVector('ψ', num_four_patch_d_and_r_repetitions)
    two_patch_2_q_pqc_parameter = ParameterVector('γ', 6 * num_two_patch_2_q_pqc_layers)

    four_patch_one_qubit_circ = FourPatchOneQ(
        first_four_patch_pv[0], first_four_patch_pv[1], first_four_patch_pv[2], first_four_patch_pv[3],
        single_patch_encoding_parameter, single_patch_d_and_r_phase_parameter, four_patch_d_and_r_phase_parameter, two_patch_2_q_pqc_parameter, to_gate=False
    )
    four_patch_one_qubit_circ.draw(output='mpl', style='bw', filename='FourPatchOneQ.png')