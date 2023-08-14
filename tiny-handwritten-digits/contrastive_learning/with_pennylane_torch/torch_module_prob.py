"""
QNNs with probability vector as output.
Wrapped for use with PyTorch.
"""
from qnns import RecurrentCircV1
import pennylane as qml

import torch
import torch.nn as nn

class RecurentQNNNoPosCodeV1(nn.Module):
    """
    PyTorch wrapped module for RecurrentCircV1.
    The input image data is assumed to be of shape (...,  64), each 16-element segment is a 4 by 4 patch.
    """
    def __init__(self, L1, L2, L_MC, n_mem_qubits=4, n_patch_qubits=4, forget_gate=False):
        super().__init__()
        wires = list(range(n_mem_qubits+ n_patch_qubits))
        dev = qml.device("default.mixed", wires=wires)
        @qml.qnode(dev, interface="torch")
        def qnn_probs(
                inputs,
                four_pixel_encode_params,
                sixteen_pixel_encode_params,
                mem_init_params,
                mem_patch_interact_params,
                mem_computation_params
        ):
            RecurrentCircV1(
                inputs,
                four_pixel_encode_params,
                sixteen_pixel_encode_params,
                mem_init_params,
                mem_patch_interact_params,
                mem_computation_params,
                wires,
                L1,
                L2,
                L_MC,
                forget_gate
            )
            return qml.probs(wires[:n_mem_qubits])
        weight_shapes = {
            "four_pixel_encode_params": (L2, L1 * 6 * 4),
            "sixteen_pixel_encode_params": (L2, 21),
            "mem_init_params": 12 * n_mem_qubits - 9,
            "mem_patch_interact_params": 36, # currently only support 2-to-2 interaction
            "mem_computation_params": (L_MC, 12 * n_mem_qubits - 9)
        }
        init_method = {
            "four_pixel_encode_params": nn.init.normal_,
            "sixteen_pixel_encode_params": nn.init.normal_,
            "mem_init_params": nn.init.normal_,
            "mem_patch_interact_params": nn.init.normal_,
            "mem_computation_params": nn.init.normal_
        }
        self.qlayer = qml.qnn.TorchLayer(qnn_probs, weight_shapes=weight_shapes, init_method=init_method)

    def forward(self, inputs):
        return self.qlayer(inputs)

