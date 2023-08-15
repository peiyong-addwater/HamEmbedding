import pennylane as qml
import torch
from torch import multiprocessing as mp
from torch import nn as nn
from torch.nn import functional as F
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass

qml.disable_return()

class TorchLayer(qml.qnn.TorchLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_cpus = mp.cpu_count()

    def forward(self, inputs):
        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        if not qml.active_return() and has_batch_dim:


            #reconstructor = [self._evaluate_qnode(x) for x in torch.unbind(inputs)]
            #results = torch.stack(reconstructor)
            results = self._evaluate_qnode(inputs)
        else:
            # calculate the forward pass as usual
            results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

