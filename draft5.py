from Models.qiskit_models import classification8x8Image10ClassesSamplerSimpleQRNN, ClassificationSamplerSimpleQRNN8x8Image
import numpy as np
import torch

n_mem = 3
n_reuploading = 2

num_mem_qubits = n_mem
num_single_patch_reuploading = n_reuploading
num_classification_layers = 1

num_classification_qubits = 4
num_total_qubits = num_mem_qubits + 3

num_single_patch_reuploading_params = 30 * num_single_patch_reuploading  # using the fourByFourPatchReuploadPooling1Q function
num_mem_params = 6 * (num_mem_qubits + 1)  # using the allInOneAnsatz function
num_classification_params = num_classification_layers * 3 * num_classification_qubits  # using the simplePQC function
num_total_params = num_single_patch_reuploading_params + num_mem_params + num_classification_params
num_backbone_params = num_single_patch_reuploading_params + num_mem_params

qnn, n_weights, n_input = classification8x8Image10ClassesSamplerSimpleQRNN(
    num_single_patch_reuploading=num_single_patch_reuploading,
    num_mem_qubits=num_mem_qubits,
    num_classification_layers=num_classification_layers
)

input_data = np.random.rand(n_input)
weights = np.random.rand(n_weights)

res = qnn.forward(input_data, weights)
input_grad, weight_grad = qnn.backward(input_data, weights)

print("QNN forward pass result:")
print(res)
print("QNN input gradient:")
print(input_grad)
print("QNN weight gradient:")
print(weight_grad)

print("="*20)

qnn_torch = ClassificationSamplerSimpleQRNN8x8Image(
    num_single_patch_reuploading=num_single_patch_reuploading,
    num_mem_qubits=num_mem_qubits,
    num_classification_layers=num_classification_layers
)

input_data = torch.Tensor(input_data)
weights = torch.Tensor(weights)

print("QNN torch forward pass result:")
print(qnn_torch(input_data)) # different than the first forward pass result since the weights are different