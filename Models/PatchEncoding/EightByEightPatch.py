from pennylane.operation import Operation, AnyWires
from SU4 import SU4, TailLessSU4, HeadlessSU4
import pennylane as qml

def pqc_64(params, wires):
    """
    A 4-qubit, 64-parameter PQC.
    :param params:
    :param wires: 4 qubit
    :return:
    """
    SU4(params[...,:15], wires=[wires[0], wires[1]])
    SU4(params[...,15:30], wires=[wires[2], wires[3]])
    HeadlessSU4(params[...,30:39], wires=[wires[1], wires[2]])
    HeadlessSU4(params[...,39:48], wires=[wires[0], wires[2]])
    HeadlessSU4(params[...,48:57], wires=[wires[1], wires[3]])
    qml.IsingXX(params[...,57], wires=[wires[0], wires[3]])
    qml.IsingYY(params[...,58], wires=[wires[0], wires[3]])
    qml.IsingZZ(params[...,59], wires=[wires[0], wires[3]])
    qml.U2(params[...,60], params[...,61], wires=wires[0])
    qml.U2(params[...,62], params[...,63], wires=wires[3])


class EightByEightDirectReUploading(Operation):
    """
    Encode 64 pixels into 4 qubits
    Pixel values are encoded using a 4-qubit, 64-parameter PQC.
    Then followed by a chain of TailLessSU4 gates, from bottom up: 3->2, 2->1, 1->0.
    Each TailLessSU4 gate has 9 parameters.
    Total number of trainable parameters: 3*9*L
    Where L is the number of re-uploading layers.
    """
    num_wires = 4
    grad_method = None

    def __init__(self, pixels, params, wires, L, id=None):
        """

        :param pixels: flattened 8 by 8 patch, shape (batch_size, 64)
        :param params: trainable parameters, shape (..., L*3*9)
        :param wires: 4
        :param L: number of data re-uploading layers
        :param id:
        """
        interface = qml.math.get_interface(params)
        params = qml.math.asarray(params, interface=interface)
        pixels = qml.math.asarray(pixels, interface=interface)
        params_shape = qml.math.shape(params)
        pixels_shape = qml.math.shape(pixels)

        if not (len(pixels_shape)==2 or len(pixels_shape)==1):
            raise ValueError(f"pixels must be a 2D or 1D array, got shape {pixels_shape}")

        if not (len(params_shape)==2 or len(params_shape)==1):
            raise ValueError(f"params must be a 2D or 1D array, got shape {params_shape}")

        if params_shape[-1] != 3*9*L:
            raise ValueError(f"params must be an array of shape (..., {3*9*L}), got {params_shape}")

        if pixels_shape[-1] != 64:
            raise ValueError(f"pixels must be an array of shape (..., 64), got {pixels_shape}")

        self._hyperparameters = {"L": L}
        super().__init__(pixels, params, wires=wires, id=id)

    @property
    def num_params(self):
        return 2

    @staticmethod
    def compute_decomposition(pixels, params, wires, L):
        op_list = []
        for i in range(L):
            op_list.append(pqc_64(pixels[...,:], wires))
            op_list.append(qml.Barrier(only_visual=True, wires=wires))
            op_list.append(TailLessSU4(params[..., 0 + 27 * i:9 + 27 * i], wires=[wires[3], wires[2]]))
            op_list.append(TailLessSU4(params[..., 9 + 27 * i:18 + 27 * i], wires=[wires[2], wires[1]]))
            op_list.append(TailLessSU4(params[..., 18 + 27 * i:27 + 27 * i], wires=[wires[1], wires[0]]))
            op_list.append(qml.Barrier(only_visual=True, wires=wires))
            op_list.append(qml.Barrier(only_visual=True, wires=wires))
        return op_list

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import torch

    dev4q = qml.device('default.qubit', wires=4)

    L = 2
    params = torch.randn(3*9*L)
    pixels = torch.randn(2,64)

    @qml.qnode(dev4q)
    def circuit(pixels, params):
        EightByEightDirectReUploading.compute_decomposition(pixels, params, wires=[0, 1, 2, 3], L=L)
        return qml.probs()

    print(circuit(pixels, params))

    fig, ax = qml.draw_mpl(circuit, style='sketch')(pixels,params)
    fig.savefig('8x8_patch_reupload.png')
    plt.close(fig)