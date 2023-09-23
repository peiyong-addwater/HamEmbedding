"""
QNNs with probability vector as output.
Wrapped for use with PyTorch.
"""
from .qnns_pennylane_v031 import RecurrentCircV1
import pennylane as qml

qml.disable_return()

import torch
import torch.nn as nn

class RecurentQNNNoPosCodeV1(nn.Module):
    """
    PyTorch wrapped module for RecurrentCircV1.
    The input image data is assumed to be of shape (...,  64).
    Every 16 pixels are considered as a patch.
    Output size is 2 ** n_mem_qubits.
    """
    def __init__(self, L1, L2, L_MC, n_mem_qubits=4, n_patch_qubits=4, forget_gate=False, diff_method='spsa'):
        super().__init__()
        wires = list(range(n_mem_qubits+ n_patch_qubits))
        dev = qml.device("default.mixed", wires=wires)
        @qml.qnode(dev, interface="torch", diff_method=diff_method)
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
        # the input will have shape (batchsize, 1, 8, 8) for training
        # for testing, after squeeze, it will have shape (1, 8, 8)
        inputs = torch.squeeze(inputs, 1) # only squeeze the channel dimension
        #inputs = inputs.reshape(inputs.shape[0], 64)
        inputs = self.cut8x8to4x4PatchesNoPos(inputs)
        return self.qlayer(inputs)

    def cut8x8to4x4PatchesNoPos(self, img: torch.Tensor):
        batchsize = img.shape[0]
        patches = torch.zeros((batchsize, 64))
        for i in range(2):
            for j in range(2):
                patches[:, 16 * (2 * i + j):16 * (2 * i + j + 1)] = img[:, 4 * i:4 * i + 4, 4 * j:4 * j + 4].flatten(
                    start_dim=1)
        return patches

if __name__ == '__main__':
    mem_qubits = 4
    patch_qubits = 4
    L1 = 2
    L2 = 2
    L_MC = 2
    data = torch.randn(4, 1, 8, 8)
    model = RecurentQNNNoPosCodeV1(L1, L2, L_MC, mem_qubits, patch_qubits)
    #print(model(data))