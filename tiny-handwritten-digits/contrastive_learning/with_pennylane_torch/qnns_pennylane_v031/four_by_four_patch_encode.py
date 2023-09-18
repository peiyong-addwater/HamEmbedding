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

# sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')


class FourPixelReUpload(Operation):
    """
    Data re-uploading layer for four pixels. Pixel data is encoded as rotation parameters of RX and RY gates;
    Entanglement is achieved by CZ gates.
    Trainable parameters are the rotation parameters of Rot gates
    """
    num_wires = 2
    grad_method = None

    def __init__(self, pixels, encode_params, L1, wires,do_queue=None, id=None):
        """

        :param pixels: input pixel data. Four element array
        :param encode_params: trainable parameters for encoding. 6*L1 element array
        :param L1: number of data re-uploading layers
        :param wires: Qubits to be used. Number of qubits: 2
        :param do_queue:
        :param id:
        """
        interface = qml.math.get_interface(encode_params)
        encode_params = qml.math.asarray(encode_params, like=interface)
        pixels = qml.math.asarray(pixels, like=interface)
        pixel_shape = qml.math.shape(pixels)
        encode_params_shape = qml.math.shape(encode_params)

        if not (len(pixel_shape)==1 or len(pixel_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"pixels must be a 1D or 2D array; got shape {pixel_shape}")
        if pixel_shape[-1]!=4:
            raise ValueError(f"pixels must be a 1D or 2D array with last dimension 4; got shape {pixel_shape}")
        if not (len(encode_params_shape)==1 or len(encode_params_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"encode_params must be a 1D or 2D array; got shape {encode_params_shape}")
        if encode_params_shape[-1]!=6*L1:
            raise ValueError(f"encode_params must be a 1D or 2D array with last dimension 6*{L1}; got shape {encode_params_shape}")
        self._hyperparameters = {"L1": L1}

        super().__init__(pixels,encode_params, wires=wires,  id=id)
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
            qml.Barrier()
        return op_list

class FourByFourPatchReUpload(Operation):
    """
    Encode 16 pixels into 4 qubits;
    The 16 pixels are divided into 4 groups, each group has 4 pixels;
    Each group of 4 pixels is encoded into 2 qubits using 'FourPixelReUpload';
    And have different parameters for each group of 4 pixels;
    Then only for the 'FourPixelReUpload', the total number of parameters is 6*L1*4 for a single layer of 'FourByFourPatchReUpload';
    Then the total parameter for four_pixel_encode_parameters should be in shape (...,L2, 6*L1*4)
    Plus a layer of Rot gates and CRot gates, giving us 4*3+3*3=21 parameters per layer of 'FourByFourPatchReUpload';
    Then the shape of sixteen_pixel_parameters should be (...,L2, 21)
    """
    num_wires = 4
    grad_method = None

    def __init__(self, pixels_len_16, four_pixel_encode_parameters, sixteen_pixel_parameters, L1, L2, wires, do_queue=None, id=None):
        interface = qml.math.get_interface(four_pixel_encode_parameters)
        four_pixel_encode_parameters = qml.math.asarray(four_pixel_encode_parameters, like=interface)
        pixels_len_16 = qml.math.asarray(pixels_len_16, like=interface)
        sixteen_pixel_parameters = qml.math.asarray(sixteen_pixel_parameters, like=interface)

        self._hyperparameters = {"L1": L1, "L2": L2}

        pixels_len_16_shape = qml.math.shape(pixels_len_16)
        four_pixel_encode_parameters_shape = qml.math.shape(four_pixel_encode_parameters)
        sixteen_pixel_parameters_shape = qml.math.shape(sixteen_pixel_parameters)
        if not (len(pixels_len_16_shape)==1 or len(pixels_len_16_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"pixels must be a 1D or 2D array; got shape {pixels_len_16_shape}")
        if pixels_len_16_shape[-1]!=16:
            raise ValueError(f"pixels must be a 1D or 2D array with last dimension 16; got shape {pixels_len_16_shape}")
        if not (len(four_pixel_encode_parameters_shape)==2 or len(four_pixel_encode_parameters_shape)==3): # 3 is when batching, 2 is not batching
            raise ValueError(f"four_pixel_encode_parameters must be a 2D or 3D array; got shape {four_pixel_encode_parameters_shape}")
        if four_pixel_encode_parameters_shape[-1]!=6*L1*4 or four_pixel_encode_parameters_shape[-2]!=L2:
            raise ValueError(f"four_pixel_encode_parameters must be a 2D or 3D array with dimension (...,{L2}, 6*{L1}*4); got shape {four_pixel_encode_parameters_shape}")
        if not (len(sixteen_pixel_parameters_shape)==2 or len(sixteen_pixel_parameters_shape)==3): # 3 is when batching, 2 is not batching
            raise ValueError(f"sixteen_pixel_parameters must be a 2D or 3D array with shape (...,{L2}, 21); got shape {sixteen_pixel_parameters_shape}")
        if sixteen_pixel_parameters_shape[-2]!=L2 or sixteen_pixel_parameters_shape[-1]!=21:
            raise ValueError(f"sixteen_pixel_parameters must be a 2D or 3D array with shape (...,{L2}, 21); got shape {sixteen_pixel_parameters_shape}")

        super().__init__(pixels_len_16, four_pixel_encode_parameters, sixteen_pixel_parameters, wires=wires,  id=id)

    @property
    def num_params(self):
        return 3

    @staticmethod
    def compute_decomposition(pixels_len_16, four_pixel_encode_parameters, sixteen_pixel_parameters, wires, L1, L2):
        op_list = []
        for i in range(L2):
            op_list.append(FourPixelReUpload(pixels_len_16[...,0:4], four_pixel_encode_parameters[...,i,0:6*L1], L1, wires=[wires[0], wires[1]]))
            op_list.append(
                FourPixelReUpload(pixels_len_16[..., 4:8], four_pixel_encode_parameters[..., i, 6 * L1:6 * L1*2], L1,
                                  wires=[wires[2], wires[3]]))
            op_list.append(qml.CZ(wires=[wires[0], wires[2]]))
            op_list.append(qml.CZ(wires=[wires[1], wires[3]]))
            op_list.append(
                FourPixelReUpload(pixels_len_16[..., 8:12], four_pixel_encode_parameters[..., i, 6 * L1*2:6 * L1 * 3], L1,
                                  wires=[wires[0], wires[1]]))
            op_list.append(
                FourPixelReUpload(pixels_len_16[..., 12:16], four_pixel_encode_parameters[..., i, 6 * L1*3:6 * L1  *4], L1,
                                  wires=[wires[2], wires[3]]))
            op_list.append(qml.CZ(wires=[wires[0], wires[2]]))
            op_list.append(qml.CZ(wires=[wires[1], wires[3]]))
            op_list.append(qml.Rot(sixteen_pixel_parameters[..., i, 0], sixteen_pixel_parameters[..., i, 1],sixteen_pixel_parameters[..., i, 2], wires=wires[0]))
            op_list.append(qml.Rot(sixteen_pixel_parameters[..., i, 3], sixteen_pixel_parameters[..., i, 4],sixteen_pixel_parameters[..., i, 5], wires=wires[1]))
            op_list.append(qml.Rot(sixteen_pixel_parameters[..., i, 6], sixteen_pixel_parameters[..., i, 7],sixteen_pixel_parameters[..., i, 8], wires=wires[2]))
            op_list.append(qml.Rot(sixteen_pixel_parameters[..., i, 9], sixteen_pixel_parameters[..., i, 10],sixteen_pixel_parameters[..., i, 11], wires=wires[3]))
            # CRot direction: 3->2, 2->1, 1->0
            op_list.append(qml.CRot(sixteen_pixel_parameters[..., i, 12], sixteen_pixel_parameters[..., i, 13],sixteen_pixel_parameters[..., i, 14], wires=[wires[3], wires[2]]))
            op_list.append(qml.CRot(sixteen_pixel_parameters[..., i, 15], sixteen_pixel_parameters[..., i, 16],sixteen_pixel_parameters[..., i, 17], wires=[wires[2], wires[1]]))
            op_list.append(qml.CRot(sixteen_pixel_parameters[..., i, 18], sixteen_pixel_parameters[..., i, 19],sixteen_pixel_parameters[..., i, 20], wires=[wires[1], wires[0]]))
            qml.Barrier()
        return op_list


class FourByFourPatchWithPosEncoding(Operation):
    """
    FourByFourPatchReUpload plus positional encoding.
    The maximum number of qubits carrying features is 3,
    since there is always one qubit carrying the pos encoding.
    The pos encoding is in the form of two integers in {0,1}, in this case,
    they will be attached at the end of the 16-pixel data vector,
    making it an 18-element vector for the input (single batch).
    The trainable parameters remain the same as 'FourByFourPatchReUpload'.
    All qubits except the feature-carrying qubits will be reset before encoding the pos information
    """
    num_wires = 4
    grad_method = None

    def __init__(self, pixels_whole_patch_with_pos, four_pixel_encode_parameters, sixteen_pixel_parameters, L1, L2, n_feature_qubits, wires, do_queue=None, id=None):
        interface = qml.math.get_interface(four_pixel_encode_parameters)
        four_pixel_encode_parameters = qml.math.asarray(four_pixel_encode_parameters, like=interface)
        pixels_whole_patch_with_pos = qml.math.asarray(pixels_whole_patch_with_pos, like=interface)
        sixteen_pixel_parameters = qml.math.asarray(sixteen_pixel_parameters, like=interface)

        self._hyperparameters = {"L1": L1, "L2": L2, "n_feature_qubits": n_feature_qubits}

        pixels_whole_patch_with_pos_shape = qml.math.shape(pixels_whole_patch_with_pos)
        four_pixel_encode_parameters_shape = qml.math.shape(four_pixel_encode_parameters)
        sixteen_pixel_parameters_shape = qml.math.shape(sixteen_pixel_parameters)
        if not (len(pixels_whole_patch_with_pos_shape)==1 or len(pixels_whole_patch_with_pos_shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError(f"pixels must be a 1D or 2D array; got shape {pixels_whole_patch_with_pos_shape}")
        if pixels_whole_patch_with_pos_shape[-1]!=18:
            raise ValueError(f"pixels must be a 1D or 2D array with last dimension 18; got shape {pixels_whole_patch_with_pos_shape}")
        if not (len(four_pixel_encode_parameters_shape)==2 or len(four_pixel_encode_parameters_shape)==3): # 3 is when batching, 2 is not batching
            raise ValueError(f"four_pixel_encode_parameters must be a 2D or 3D array; got shape {four_pixel_encode_parameters_shape}")
        if four_pixel_encode_parameters_shape[-1]!=6*L1*4 or four_pixel_encode_parameters_shape[-2]!=L2:
            raise ValueError(f"four_pixel_encode_parameters must be a 2D or 3D array with dimension (...,{L2}, 6*{L1}*4); got shape {four_pixel_encode_parameters_shape}")
        if not (len(sixteen_pixel_parameters_shape)==2 or len(sixteen_pixel_parameters_shape)==3): # 3 is when batching, 2 is not batching
            raise ValueError(f"sixteen_pixel_parameters must be a 2D or 3D array with shape (...,{L2}, 21); got shape {sixteen_pixel_parameters_shape}")
        if sixteen_pixel_parameters_shape[-2]!=L2 or sixteen_pixel_parameters_shape[-1]!=21:
            raise ValueError(f"sixteen_pixel_parameters must be a 2D or 3D array with shape (...,{L2}, 21); got shape {sixteen_pixel_parameters_shape}")
        if n_feature_qubits>=4:
            raise ValueError(f"The number of feature qubits should not be more than 3, got {n_feature_qubits}")

        super().__init__(pixels_whole_patch_with_pos, four_pixel_encode_parameters, sixteen_pixel_parameters, L1, L2, n_feature_qubits, wires=wires,  id=id)

    @property
    def num_params(self):
        return 3

    @staticmethod
    def compute_decomposition(pixels_whole_patch_with_pos, four_pixel_encode_parameters, sixteen_pixel_parameters, wires, L1, L2, n_feature_qubits):
        op_list = []
        pos_encoding_wire = wires[n_feature_qubits]
        reset_wires = wires[n_feature_qubits:]
        data_pixels = pixels_whole_patch_with_pos[...,:-2]
        pos_encoding = pixels_whole_patch_with_pos[...,-2:]
        op_list.append(
            FourByFourPatchReUpload(data_pixels, four_pixel_encode_parameters, sixteen_pixel_parameters, L1, L2, wires)
        )
        op_list.append(ResetZeroState(wires=reset_wires))
        op_list.append(qml.RX(pos_encoding[...,0]*(math.pi), wires=pos_encoding_wire))
        op_list.append(qml.RY(pos_encoding[...,1]*(math.pi), wires=pos_encoding_wire))
        return op_list











if __name__ == '__main__':
    """
    default.mixed device does not support broadcasting
    """
    from scipy.stats import unitary_group
    import matplotlib.pyplot as plt

    qml.disable_return() # Turn of the experimental return feature,
    # see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

    dev2q = qml.device('default.mixed', wires=2)
    dev4q = qml.device('default.mixed', wires=4, shots=10)
    # print(dev2q.capabilities()["supports_broadcasting"]) # False for default.mixed

    L1 = 2
    L2 = 2

    @qml.qnode(dev2q, interface='torch')
    def four_pixel_encode(pixels, encode_params, L1, expand=False):
        if not expand:
            FourPixelReUpload(pixels, encode_params, L1, wires=[0, 1])
        else:
            FourPixelReUpload.compute_decomposition(pixels, encode_params, [0, 1], L1)
        return qml.probs()

    @qml.qnode(dev4q, interface='torch')
    def patch_encode(inputs, four_pix_params, sixteen_reupload_params):
        #FourByFourPatchReUpload(inputs, four_pix_params, sixteen_reupload_params, L1, L2, [0,1,2,3])
        FourByFourPatchReUpload.compute_decomposition(inputs, four_pix_params, sixteen_reupload_params, [0,1,2,3], L1, L2)
        #ResetZeroState([0,1,2,3])
        return qml.probs()


    #@qml.shadows.shadow_state(wires=[0, 1], diffable=True)
    @qml.qnode(dev4q, interface='torch')
    def patch_encode2(inputs, four_pix_params, sixteen_reupload_params):
        FourByFourPatchWithPosEncoding.compute_decomposition(inputs, four_pix_params, sixteen_reupload_params, [0,1,2,3], L1, L2, 2)
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    pixels = torch.randn(3,4)
    pixels_16 = torch.randn(3,16)
    pixels_18 = torch.randn(3,18)
    four_pixel_encode_params = torch.randn(L1*6)

    patch_encode_params = torch.randn(L2, L1*6*4)
    patch_rot_crot_params = torch.randn(L2, 21)


    #print(four_pixel_encode(pixels[0], four_pixel_encode_params, L1))
    fig, ax = qml.draw_mpl(four_pixel_encode, style='sketch')(pixels[0], four_pixel_encode_params, L1, True)
    fig.savefig('four_pixel_reupload.png')
    plt.close(fig)

    #print(patch_encode(pixels_16[0], patch_encode_params, patch_rot_crot_params))
    fig, ax = qml.draw_mpl(patch_encode, style='sketch')(pixels_16[0], patch_encode_params, patch_rot_crot_params)
    fig.savefig('four_by_four_patch_reupload.png')
    plt.close(fig)

    print(patch_encode2(pixels_18[0], patch_encode_params, patch_rot_crot_params))
    fig, ax = qml.draw_mpl(patch_encode2, style='sketch')(pixels_18[0], patch_encode_params, patch_rot_crot_params)
    fig.savefig('four_by_four_patch_reupload_with_pos_encoding.png')
    plt.close(fig)





    # let's try to convert the qnode into a pytorch layer
    weight_shapes = {
        "four_pix_params": (L2,L1*6*4),
        "sixteen_reupload_params": (L2, 21),
    }
    qlayer = qml.qnn.TorchLayer(patch_encode2, weight_shapes)
    clayer = torch.nn.Linear(16, 2)
    model = torch.nn.Sequential(qlayer)
    print(model(pixels_18))