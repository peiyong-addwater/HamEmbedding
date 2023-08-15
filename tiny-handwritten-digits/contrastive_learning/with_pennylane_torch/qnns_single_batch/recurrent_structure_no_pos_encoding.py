import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from .reset_gate import ResetZeroState
import math
from .four_by_four_patch_encode import FourByFourPatchReUpload
from .su4 import SU4

# sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

class InitialiseMemState(Operation):
    """
    Initial state of the memory qubits. Trainable.
    Start with a layer of H gates,
    then a layer of U3 gates,
    then a chain of non-leading SU4 gates.
    Total number of parameters: 3 * num_wires + 9 * (num_wires-1)
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, parmas, wires, do_queue=None, id=None):
        """

        :param parmas: shape of (..., 3 * num_wires + 9 * (num_wires-1)= 12 * num_wires - 9)
        :param wires:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(parmas)

        super().__init__(parmas, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires):
        n_wires = len(wires)
        u3_params = params[..., :3 * n_wires]
        su4_params = params[..., 3 * n_wires:]
        op_list = []
        for wire in wires:
            op_list.append(qml.Hadamard(wire))

        for i in range(n_wires):
            op_list.append(qml.U3(*u3_params[..., 3 * i: 3 * (i + 1)], wires=wires[i]))

        for i in range(n_wires - 1):
            op_list.append(SU4(su4_params[..., 9 * i: 9 * (i + 1)], wires=[wires[i], wires[i + 1]], leading_gate = False))

        return op_list

class MemPatchInteract2to2(Operation):
    """
    Interaction between two of the memory qubits and two patch feature qubits,
    Composed of non-leading SU4 gates.
    Let the first two qubits be the memory qubits, and the last two be the patch qubits.
    Interaction map: (0,2), (1,3), (0,3), (1,2)
    Meaning there will be 4 SU4 gates.-> 4*9 = 36 parameters
    After the SU4 gates, there are also four CZ gates following the interaction map.
    """
    num_wires = 4
    grad_method = None

    def __init__(self, params, wires,do_queue=None, id=None):
        """

        :param params:
        :param wires:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)

        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires):
        op_list = []
        op_list.append(SU4(params[..., :9], wires=[wires[0], wires[2]], leading_gate=False))
        op_list.append(SU4(params[..., 9:18], wires=[wires[1], wires[3]], leading_gate=False))
        op_list.append(SU4(params[..., 18:27], wires=[wires[0], wires[3]], leading_gate=False))
        op_list.append(SU4(params[..., 27:36], wires=[wires[1], wires[2]], leading_gate=False))
        op_list.append(qml.CZ(wires=[wires[0], wires[2]]))
        op_list.append(qml.CZ(wires=[wires[1], wires[3]]))
        op_list.append(qml.CZ(wires=[wires[0], wires[3]]))
        op_list.append(qml.CZ(wires=[wires[1], wires[2]]))
        return op_list

class MemComputation(Operation):
    """
    Computation on the memory qubits.
    0- Starting with n_wire - 1 CZ gates from bottom up,
    1- then a layer of U3 gates,
    2- then a chain of non-leading SU4 gates, from bottom up,
    3- then ends with a chain of CZ gates from bottom up.
    Steps 1,2,3 are repeated L_MC times.
    total number of parameters per layer: 3 * num_wires + 9 * (num_wires-1)
    parameter shape: (..., L_MC, 3 * num_wires + 9 * (num_wires-1))
    """
    num_wires = AnyWires
    grad_method = None

    def __init__(self, params, wires, L_MC, do_queue=None, id=None):
        """

        :param params: shape of (..., L_MC*( 3 * num_wires + 9 * (num_wires-1)))
        :param wires:
        :param L_MC:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        self.n_single_layer_params = 3 * n_wires + 9 * (n_wires - 1)


        self._hyperparameters = {"L_MC": L_MC}
        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires, L_MC):
        n_wires = len(wires)
        n_single_layer_params = 3 * n_wires + 9 * (n_wires - 1)
        op_list = []
        for i in range(1, n_wires):
            op_list.append(qml.CZ(wires=[wires[-i], wires[-i-1]]))
        for i in range(L_MC):
            layer_params = params[...,i*n_single_layer_params:(i+1)*n_single_layer_params]
            u3_params = layer_params[..., :3 * n_wires]
            su4_params = layer_params[..., 3 * n_wires:]
            for i in range(n_wires):
                op_list.append(qml.U3(*u3_params[..., 3 * i: 3 * (i + 1)], wires=wires[i]))
            for i in range(1, n_wires):
                op_list.append(SU4(su4_params[..., 9 * (i-1): 9 * i], wires=[wires[-i], wires[-i-1]], leading_gate=False))
            for i in range(1, n_wires):
                op_list.append(qml.CZ(wires=[wires[-i], wires[-i-1]]))
        return op_list


class RecurrentCircV1(Operation):
    """
    Recurrent circ with the following structure:
    1- Initialise the memory qubits.
    2- Encode the 4x4 patch to the bottom 4 qubits with 'FourByFourPatchReUpload'.
    3- Interaction between the last two of memory qubits and the first two patch qubits with 'MemPatchInteract2to2'.
    4- Reset the last two patch qubits to zero state.
    5- Computation on the memory qubits.
    6- Reset the first two patch qubits to zero state.
    7- (Optional) Reset the first memory qubit to zero state.
    Repeat 2-7 for 2 by 2 = 4 times for a 8 by 8 images with 4 by 4 patch size.
    The input image data is assumed to be of shape (...,  64), each 16-element segment is a 4 by 4 patch.
    structural parameters: L1, L2, L_MC
    input parameters (including the data):
        data, four_pixel_encode_parameters, sixteen_pixel_parameters, mem_init_params, mem_patch_interact_params,
        mem_computation_params, wires(total), L1, L2, L_MC, Optional_reset_first_mem_qubit
    """
    num_wires = 4+4
    grad_method = None # 'A'

    def __init__(
            self,
            data,
            four_pixel_encode_parameters,
            sixteen_pixel_parameters,
            mem_init_params,
            mem_patch_interact_params,
            mem_computation_params,
            wires,
            L1,
            L2,
            L_MC,
            Optional_reset_first_mem_qubit=False,
            do_queue=None,
            id=None
            ):
        """

        :param data: of shape (...,  64), each 16-element segment is a 4 by 4 patch.
        :param four_pixel_encode_parameters:
        :param sixteen_pixel_parameters:
        :param mem_init_params:
        :param mem_patch_interact_params:
        :param mem_computation_params:
        :param wires:
        :param L1:
        :param L2:
        :param L_MC:
        :param Optional_reset_first_mem_qubit:
        """
        interface = qml.math.get_interface(four_pixel_encode_parameters)
        data = qml.math.asarray(data, like=interface)
        four_pixel_encode_parameters = qml.math.asarray(four_pixel_encode_parameters, like=interface)
        sixteen_pixel_parameters = qml.math.asarray(sixteen_pixel_parameters, like=interface)
        mem_init_params = qml.math.asarray(mem_init_params, like=interface)
        mem_patch_interact_params = qml.math.asarray(mem_patch_interact_params, like=interface)
        mem_computation_params = qml.math.asarray(mem_computation_params, like=interface)

        data_shape = qml.math.shape(data)


        self._hyperparameters = {"L1": L1, "L2": L2, "L_MC": L_MC, "Optional_reset_first_mem_qubit": Optional_reset_first_mem_qubit}
        super().__init__(
            data,
            four_pixel_encode_parameters,
            sixteen_pixel_parameters,
            mem_init_params,
            mem_patch_interact_params,
            mem_computation_params,
            wires=wires,
            do_queue=do_queue,
            id=id,
        )

    @property
    def num_params(self):
        return 6

    @staticmethod
    def compute_decomposition(
            data,
            four_pixel_encode_parameters,
            sixteen_pixel_parameters,
            mem_init_params,
            mem_patch_interact_params,
            mem_computation_params,
            wires,
            L1,
            L2,
            L_MC,
            Optional_reset_first_mem_qubit
    ):
        op_list = []
        MB = wires[:4]
        PE = wires[4:]
        op_list.append(InitialiseMemState(mem_init_params, wires=MB))
        op_list.append(qml.Barrier())
        for i in range(4):
            # encode the 4x4 patch to the bottom 4 qubits
            op_list.append(FourByFourPatchReUpload(
                data[..., 16*i:16*(i+1)],
                four_pixel_encode_parameters,
                sixteen_pixel_parameters,
                L1,
                L2,
                wires=PE))
            op_list.append(qml.Barrier())
            # Reset the last two patch qubits to zero state.
            op_list.append(ResetZeroState(wires=PE[2:]))
            op_list.append(qml.Barrier())
            # Interaction between the last two of memory qubits and the first two patch qubits
            op_list.append(MemPatchInteract2to2(mem_patch_interact_params, wires=MB[4-2:]+PE[:2]))
            op_list.append(qml.Barrier())
            # Reset the first two patch qubits to zero state.
            op_list.append(ResetZeroState(wires=PE[:2]))
            op_list.append(qml.Barrier())
            # Computation on the memory qubits.
            op_list.append(MemComputation(mem_computation_params, wires=MB, L_MC=L_MC))
            op_list.append(qml.Barrier())
            # Reset the first memory qubit to zero state.
            if Optional_reset_first_mem_qubit:
                op_list.append(ResetZeroState(wires=MB[0]))
                op_list.append(qml.Barrier())
        return op_list


