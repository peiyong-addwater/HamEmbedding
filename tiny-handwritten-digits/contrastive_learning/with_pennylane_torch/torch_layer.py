import pennylane as qml
import torch
from torch import multiprocessing as mp
from torch import nn as nn
from torch.nn import functional as F
from pennylane.transforms.batch_transform import batch_transform
from pennylane.transforms.batch_params import _nested_stack
from pennylane import numpy as np
from pennylane.tape import QuantumTape
from typing import Callable, Sequence, Tuple, Union

qml.disable_return()



@batch_transform
def batch_input(
    tape: Union[QuantumTape, qml.QNode],
    argnum: Union[Sequence[int], int],
) -> Tuple[Sequence[QuantumTape], Callable]:

    argnum = tuple(argnum) if isinstance(argnum, (list, tuple)) else (int(argnum),)

    all_parameters = tape.get_parameters(trainable_only=False)
    print(len(all_parameters))
    argnum_params = [all_parameters[i] for i in argnum]

    if any(num in tape.trainable_params for num in argnum):
        if (qml.math.get_interface(*argnum_params) != "jax" and qml.math.get_interface(*argnum_params) != "torch"):
            raise ValueError(
                "Batched inputs must be non-trainable. Please make sure that the parameters indexed by "
                + "'argnum' are not marked as trainable."
            )


    batch_dims = np.unique([qml.math.shape(x)[0] for x in argnum_params])
    if len(batch_dims) != 1:
        raise ValueError(
            "Batch dimension for all gate arguments specified by 'argnum' must be the same."
        )

    batch_size = batch_dims[0]

    outputs = []
    for i in range(batch_size):
        batch = []
        for idx, param in enumerate(all_parameters):
            if idx in argnum:
                param = param[i]
            batch.append(param)
        outputs.append(batch)

    # Construct new output tape with unstacked inputs
    output_tapes = []
    for params in outputs:
        new_tape = tape.copy(copy_operations=True)
        new_tape.set_parameters(params, trainable_only=False)
        output_tapes.append(new_tape)

    def processing_fn(res):
        if qml.active_return():
            return _nested_stack(res)

        return qml.math.squeeze(qml.math.stack(res))

    return output_tapes, processing_fn


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
            results = self._evaluate_batch_qnode(inputs)
        else:
            # calculate the forward pass as usual
            results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

    def _evaluate_batch_qnode(self, x):
        """Evaluates the QNode for a batch of input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        batched_qnode = batch_input(self.qnode, argnum=1)
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = batched_qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        if len(x.shape) > 1:
            res = [torch.reshape(r, (x.shape[0], -1)) for r in res]

        return torch.hstack(res).type(x.dtype)

