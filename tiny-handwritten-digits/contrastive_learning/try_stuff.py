from with_pennylane_torch.qnns_single_batch import RecurrentCircV1
import pennylane as qml
from with_pennylane_torch.torch_layer import TorchLayer, batch_input
import torch


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    qml.disable_return()  # Turn of the experimental return feature,
    # see https://docs.pennylane.ai/en/stable/code/api/pennylane.enable_return.html#pennylane.enable_return

    mem_qubits = 4
    wires = list(range(mem_qubits+4))
    devfull = qml.device('default.mixed', wires=mem_qubits+4)

    mem_init_params = torch.randn( 12 * mem_qubits - 9)
    mem_patch_interact_params = torch.randn(36)
    L_MC = 1
    mem_computation_params = torch.randn(L_MC, 12 * mem_qubits - 9)
    L1 = 1
    L2 = 1
    data = torch.randn(4, 64)
    patch_encode_params = torch.randn(L2, L1 * 6 * 4)
    patch_rot_crot_params = torch.randn(L2, 21)

    @batch_input(argnum=1)
    @qml.qnode(devfull, interface="torch", diff_method='spsa')
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
            True
        )
        return qml.probs(wires[:mem_qubits])

    print(qnn_probs(data, patch_encode_params, patch_rot_crot_params, mem_init_params, mem_patch_interact_params, mem_computation_params))


