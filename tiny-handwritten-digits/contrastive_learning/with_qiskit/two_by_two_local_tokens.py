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
from token_mixing_layers import FourQubitParameterisedLayer

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