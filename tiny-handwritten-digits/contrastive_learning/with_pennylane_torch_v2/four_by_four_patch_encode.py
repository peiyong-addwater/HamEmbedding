import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')


class FourPixelReUpload(Operation):
    num_wires = 2
    grad_method = None

    def __init__(self, pixels, encode_params, L1, wires,do_queue=None, id=None):
        interface = qml.math.get_interface(encode_params)
        encode_params = qml.math.asarray(encode_params, like=interface)
        pixels = qml.math.asarray(pixels, like=interface)
        pixel_shape = qml.math.shape(pixels)
        encode_params_shape = qml.math.shape(encode_params)
        #self.wires = wires
        if not (len(pixel_shape)==1 or len(pixel_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"pixels must be a 1D or 2D array; got shape {pixel_shape}")
        if pixel_shape[-1]!=4:
            raise ValueError(f"pixels must be a 1D or 2D array with last dimension 4; got shape {pixel_shape}")
        if not (len(encode_params_shape)==1 or len(encode_params_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"encode_params must be a 1D or 2D array; got shape {encode_params_shape}")
        if encode_params_shape[-1]!=6*L1:
            raise ValueError(f"encode_params must be a 1D or 2D array with last dimension 6*L1; got shape {encode_params_shape}")
        self._hyperparameters = {"L1": L1}


        #self.pixels = pixels
        #self.encode_params = encode_params
        #self.L1 = L1
        #self.wires = wires
        super().__init__(pixels,encode_params, wires=wires, do_queue=do_queue, id=id)
    @property
    def num_params(self):
        return 2
    @staticmethod
    def compute_decomposition(pixels, encode_params, wires, L1):
        op_list = []
        for i in range(L1):
            op_list.append(qml.RX(pixels[...,0], wires=wires[0]))
            op_list.append(qml.RY(pixels[...,1], wires=wires[0]))
            op_list.append(qml.RX(pixels[...,2], wires=wires[1]))
            op_list.append(qml.RY(pixels[...,3], wires=wires[1]))
            op_list.append(qml.CZ(wires=[wires[0], wires[1]]))
            op_list.append(qml.Rot(encode_params[...,6 * i], encode_params[...,6 * i + 1], encode_params[...,6 * i + 2], wires=wires[0]))
            op_list.append(qml.Rot(encode_params[...,6 * i + 3], encode_params[...,6 * i + 4], encode_params[...,6 * i + 5], wires=wires[1]))
            op_list.append(qml.CZ(wires=[wires[0], wires[1]]))
            #qml.Barrier()


if __name__ == '__main__':
    from scipy.stats import unitary_group
    dev2q = qml.device('default.qubit', wires=2)
    dev4q = qml.device('default.qubit', wires=4)
    @qml.qnode(dev2q, interface='torch')
    def four_pixel_encode(pixels, encode_params, L1, expand=False):
        if not expand:
            FourPixelReUpload(pixels, encode_params, L1, wires=[0, 1])
        else:
            FourPixelReUpload.compute_decomposition(pixels, encode_params, [0, 1], L1)
        return qml.probs()

    pixels = torch.randn(3,4)
    L1 = 2
    encode_params = torch.randn(L1*6)



    print(four_pixel_encode(pixels, encode_params, L1))
    fig, ax = qml.draw_mpl(four_pixel_encode, style='sketch')(pixels[0], encode_params, L1, True)
    fig.savefig('four_pixel_reupload.png')