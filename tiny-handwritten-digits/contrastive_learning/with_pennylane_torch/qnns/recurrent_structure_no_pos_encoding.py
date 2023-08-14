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
    grad_method = 'A'

    def __init__(self, parmas, wires, do_queue=None, id=None):
        """

        :param parmas: shape of (..., 3 * num_wires + 9 * (num_wires-1)= 12 * num_wires - 9)
        :param wires:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(parmas)
        if not (len(params_shape)==1 or len(params_shape)==2): # 2 when is batching, 1 when not
            raise ValueError(f"params must be a 1D or 2D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9:
            raise ValueError(f"params must be an array of shape (..., 12 * {n_wires} - 9), got {params_shape}")
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
    grad_method = 'A'

    def __init__(self, params, wires,do_queue=None, id=None):
        """

        :param params:
        :param wires:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        if not (len(params_shape)==1 or len(params_shape)==2): # 2 when is batching, 1 when not
            raise ValueError(f"params must be a 1D or 2D array, got shape {params_shape}")
        if n_wires != 4:
            raise ValueError(f"num_wires must be 4, got {n_wires}")
        if params_shape[-1] != 36:
            raise ValueError(f"params must be an array of shape (..., 36), got {params_shape}")
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
    grad_method = 'A'

    def __init__(self, params, wires, L_MC, do_queue=None, id=None):
        """

        :param params: shape of (..., L_MC, 3 * num_wires + 9 * (num_wires-1))
        :param wires:
        :param L_MC:
        :param do_queue:
        :param id:
        """
        n_wires = len(wires)
        params_shape = qml.math.shape(params)
        if not (len(params_shape)==2 or len(params_shape)==3):
            raise ValueError(f"params must be a 2D or 3D array, got shape {params_shape}")
        if params_shape[-1] != 12 * n_wires - 9 or params_shape[-2] != L_MC:
            raise ValueError(f"params must be an array of shape (...,{L_MC}, 12 * {n_wires} - 9), got {params_shape}")

        self._hyperparameters = {"L_MC": L_MC}
        super().__init__(params, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(params, wires, L_MC):
        n_wires = len(wires)
        op_list = []
        for i in range(1, n_wires):
            op_list.append(qml.CZ(wires=[wires[-i], wires[-i-1]]))
        for i in range(L_MC):
            layer_params = params[...,i,:]
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
    grad_method = 'A'

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

        :param data:
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
        if not (len(data_shape)==2 or len(data_shape)==1): # 2 when is batching, 1 when not
            raise ValueError(f"data must be a 2D or 3D array, got shape {data_shape}")
        if data_shape[-1] != 64 :
            raise ValueError(f"data must be an array of shape (..., 64), got {data_shape}")

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.disable_return()  # Turn of the experimental return feature,
    # see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return
    mem_qubits = 4
    dev_mem = qml.device('default.mixed', wires=mem_qubits)
    dev4q = qml.device('default.mixed', wires=4)
    devfull = qml.device('default.mixed', wires=mem_qubits+4)

    mem_init_params = torch.randn( 12 * mem_qubits - 9)
    mem_patch_interact_params = torch.randn(36)
    L_MC = 2
    mem_computation_params = torch.randn(L_MC, 12 * mem_qubits - 9)
    L1 = 2
    L2 = 2
    data = torch.randn(4, 16)
    patch_encode_params = torch.randn(L2, L1 * 6 * 4)
    patch_rot_crot_params = torch.randn(L2, 21)

    @qml.qnode(devfull)
    def rec_circ_v1():
        RecurrentCircV1.compute_decomposition(
            data,
            patch_encode_params,
            patch_rot_crot_params,
            mem_init_params,
            mem_patch_interact_params,
            mem_computation_params,
            wires=list(range(mem_qubits+4)),
            L1=L1,
            L2=L2,
            L_MC=L_MC,
            Optional_reset_first_mem_qubit=False
        )
        return qml.probs(wires=range(mem_qubits))

    print(rec_circ_v1().shape)
    fig, ax = qml.draw_mpl(rec_circ_v1, style = 'sketch')()
    fig.savefig('rec_circ_v1.png')


    @qml.qnode(dev_mem)
    def mem_init(params):
        InitialiseMemState.compute_decomposition(params, wires=range(mem_qubits))
        return qml.probs()

    print(mem_init(mem_init_params).shape)
    fig, ax = qml.draw_mpl(mem_init, style = 'sketch')(mem_init_params)
    fig.savefig('mem_init.png')
    plt.close(fig)

    @qml.qnode(dev4q)
    def mem_patch_interact(params):
        MemPatchInteract2to2.compute_decomposition(params, wires=range(4))
        return qml.probs()

    print(mem_patch_interact(mem_patch_interact_params).shape)
    fig, ax = qml.draw_mpl(mem_patch_interact, style = 'sketch')(mem_patch_interact_params)
    fig.savefig('mem_patch_interact.png')
    plt.close(fig)

    @qml.qnode(dev_mem)
    def mem_computation(params):
        MemComputation.compute_decomposition(params, wires=range(mem_qubits), L_MC=L_MC)
        return qml.probs()

    print(mem_computation(mem_computation_params).shape)
    fig, ax = qml.draw_mpl(mem_computation, style = 'sketch', expand='device')(mem_computation_params)
    fig.savefig('mem_computation.png')





