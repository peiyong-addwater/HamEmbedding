import pennylane as qml
import torch
from pennylane.operation import Operation, AnyWires

class FourPixelReUpload(Operation):
    """
    Data re-uploading layer for four pixels. Pixel data is encoded as rotation parameters of RX and RY gates;
    Entanglement is achieved by CZ gates.
    Trainable parameters are the rotation parameters of Rot gates
    """
    num_wires = 2
    grad_method = None

    def __init__(self, pixels, encode_params, L1, wires, id=None):
        """

        :param pixels: input pixel data. Four element array
        :param encode_params:  trainable parameters for encoding. 6*L1 element array
        :param L1: number of data re-uploading layers
        :param wires: Qubits to be used. Number of qubits: 2
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
            op_list.append(qml.RX(pixels[..., 0], wires=wires[0]))
            op_list.append(qml.RY(pixels[..., 1], wires=wires[0]))
            op_list.append(qml.RX(pixels[..., 2], wires=wires[1]))
            op_list.append(qml.RY(pixels[..., 3], wires=wires[1]))
            op_list.append(qml.CZ(wires=[wires[0], wires[1]]))
            op_list.append(
                qml.Rot(encode_params[..., 6 * i], encode_params[..., 6 * i + 1], encode_params[..., 6 * i + 2],
                        wires=wires[0]))
            op_list.append(
                qml.Rot(encode_params[..., 6 * i + 3], encode_params[..., 6 * i + 4], encode_params[..., 6 * i + 5],
                        wires=wires[1]))
            op_list.append(qml.CZ(wires=[wires[0], wires[1]]))
            op_list.append(qml.Barrier(only_visual=True, wires=wires))
        return op_list

class FourPixelAmpEnc(Operation):
    """
    Encode four pixel values with Amplitude encoding.
    Only to encode the pixels into real amplitudes, no trainable parameters.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, pixels, wires, id=None):
        """

        :param pixels: input pixel data. Four element array
        :param wires: Qubits to be used. Number of qubits: 2
        :param id:
        """
        interface = qml.math.get_interface(pixels)
        pixels = qml.math.asarray(pixels, like=interface)
        pixel_shape = qml.math.shape(pixels)

        if not (len(pixel_shape) == 1 or len(pixel_shape) == 2):  # 2 is when batching, 1 is not batching
            raise ValueError(f"pixels must be a 1D or 2D array; got shape {pixel_shape}")
        if pixel_shape[-1] != 4:
            raise ValueError(f"pixels must be a 1D or 2D array with last dimension 4; got shape {pixel_shape}")

        super().__init__(pixels, wires=wires, id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(pixels, wires):
        op_list = []
        op_list.append(qml.AmplitudeEmbedding(pixels[...,:], wires=wires))
        op_list.append(qml.Barrier(only_visual=True, wires=wires))
        return op_list