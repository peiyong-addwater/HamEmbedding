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
import functools
import inspect
import math
from collections.abc import Iterable
from typing import Callable, Dict, Union, Any, Text

import pennylane as qml
from pennylane.qnode import QNode

try:
    import torch
    from torch.nn import Module

    TORCH_IMPORTED = True
except ImportError:
    # The following allows this module to be imported even if PyTorch is not installed. Users
    # will instead see an ImportError when instantiating the TorchLayer.
    from unittest.mock import Mock

    Module = Mock
    TORCH_IMPORTED = False

qml.disable_return()



@batch_transform
def batch_input(
    tape: Union[QuantumTape, qml.QNode],
    argnum: Union[Sequence[int], int],
) -> Tuple[Sequence[QuantumTape], Callable]:

    argnum = tuple(argnum) if isinstance(argnum, (list, tuple)) else (int(argnum),)

    all_parameters = tape.get_parameters(trainable_only=False)
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
                param = param[i,:]
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


class TorchLayer(Module):
    def __init__(
        self,
        qnode: QNode,
        weight_shapes: dict,
        init_method: Union[Callable, Dict[str, Union[Callable, Any]]] = None,
        # FIXME: Cannot change type `Any` to `torch.Tensor` in init_method because it crashes the
        # tests that don't use torch module.
    ):
        if not TORCH_IMPORTED:
            raise ImportError(
                "TorchLayer requires PyTorch. PyTorch can be installed using:\n"
                "pip install torch\nAlternatively, "
                "visit https://pytorch.org/get-started/locally/ for detailed "
                "instructions."
            )
        super().__init__()

        weight_shapes = {
            weight: (tuple(size) if isinstance(size, Iterable) else () if size == 1 else (size,))
            for weight, size in weight_shapes.items()
        }

        # validate the QNode signature, and convert to a Torch QNode.
        # TODO: update the docstring regarding changes to restrictions when tape mode is default.
        self._signature_validation(qnode, weight_shapes)
        self.qnode = qnode
        self.qnode.interface = "torch"

        self.qnode_weights: Dict[str, torch.nn.Parameter] = {}

        self._init_weights(init_method=init_method, weight_shapes=weight_shapes)
        self._initialized = True

    def _signature_validation(self, qnode: QNode, weight_shapes: dict):
        sig = inspect.signature(qnode.func).parameters

        if self.input_arg not in sig:
            raise TypeError(
                f"QNode must include an argument with name {self.input_arg} for inputting data"
            )

        if self.input_arg in set(weight_shapes.keys()):
            raise ValueError(
                f"{self.input_arg} argument should not have its dimension specified in "
                f"weight_shapes"
            )

        param_kinds = [p.kind for p in sig.values()]

        if inspect.Parameter.VAR_POSITIONAL in param_kinds:
            raise TypeError("Cannot have a variable number of positional arguments")

        if inspect.Parameter.VAR_KEYWORD not in param_kinds and set(weight_shapes.keys()) | {
            self.input_arg
        } != set(sig.keys()):
            raise ValueError("Must specify a shape for every non-input parameter in the QNode")

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """Evaluates a forward pass through the QNode based upon input data and the initialized
        weights.

        Args:
            inputs (tensor): data to be processed

        Returns:
            tensor: output data
        """
        has_batch_dim = len(inputs.shape) > 1

        # in case the input has more than one batch dimension
        if has_batch_dim:
            batch_dims = inputs.shape[:-1]
            inputs = torch.reshape(inputs, (-1, inputs.shape[-1]))

        if not qml.active_return() and has_batch_dim:
            # If the input has a batch dimension and we want to execute each data point separately,
            # unstack the input along its first dimension, execute the QNode on each of the yielded
            # tensors, and then stack the outputs back into the correct shape
            reconstructor = [self._evaluate_qnode(x) for x in torch.unbind(inputs)]
            results = torch.stack(reconstructor)
        else:
            # calculate the forward pass as usual
            results = self._evaluate_qnode(inputs)

        # reshape to the correct number of batch dims
        if has_batch_dim:
            results = torch.reshape(results, (*batch_dims, *results.shape[1:]))

        return results

    def _evaluate_qnode(self, x):
        """Evaluates the QNode for a single input datapoint.

        Args:
            x (tensor): the datapoint

        Returns:
            tensor: output datapoint
        """
        kwargs = {
            **{self.input_arg: x},
            **{arg: weight.to(x) for arg, weight in self.qnode_weights.items()},
        }
        res = self.qnode(**kwargs)

        if isinstance(res, torch.Tensor):
            return res.type(x.dtype)

        if len(x.shape) > 1:
            res = [torch.reshape(r, (x.shape[0], -1)) for r in res]

        return torch.hstack(res).type(x.dtype)

    def construct(self, args, kwargs):
        """Constructs the wrapped QNode on input data using the initialized weights.

        This method was added to match the QNode interface. The provided args
        must contain a single item, which is the input to the layer. The provided
        kwargs is unused.

        Args:
            args (tuple): A tuple containing one entry that is the input to this layer
            kwargs (dict): Unused
        """
        x = args[0]
        kwargs = {
            self.input_arg: x,
            **{arg: weight.data.to(x) for arg, weight in self.qnode_weights.items()},
        }
        self.qnode.construct((), kwargs)

    def __getattr__(self, item):
        """If the given attribute does not exist in the class, look for it in the wrapped QNode."""
        if self._initialized:
            return getattr(self.qnode, item)

        try:
            return self.__dict__[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, item, val):
        """If the given attribute does not exist in the class, try to set it in the wrapped QNode."""
        if self._initialized:
            setattr(self.qnode, item, val)
        else:
            self.__dict__[item] = val

    def _init_weights(
        self,
        weight_shapes: Dict[str, tuple],
        init_method: Union[Callable, Dict[str, Union[Callable, Any]], None],
    ):
        r"""Initialize and register the weights with the given init_method. If init_method is not
        specified, weights are randomly initialized from the uniform distribution on the interval
        [0, 2π].

        Args:
            weight_shapes (dict[str, tuple]): a dictionary mapping from all weights used in the QNode to
                their corresponding shapes
            init_method (Union[Callable, Dict[str, Union[Callable, torch.Tensor]], None]): Either a
                `torch.nn.init <https://pytorch.org/docs/stable/nn.init.html>`__ function for
                initializing the QNode weights or a dictionary specifying the callable/value used for
                each weight. If not specified, weights are randomly initialized using the uniform
                distribution over :math:`[0, 2 \pi]`.
        """

        def init_weight(weight_name: str, weight_size: tuple) -> torch.Tensor:
            """Initialize weights.

            Args:
                weight_name (str): weight name
                weight_size (tuple): size of the weight

            Returns:
                torch.Tensor: tensor containing the weights
            """
            if init_method is None:
                init = functools.partial(torch.nn.init.uniform_, b=2 * math.pi)
            elif callable(init_method):
                init = init_method
            elif isinstance(init_method, dict):
                init = init_method[weight_name]
                if isinstance(init, torch.Tensor):
                    if tuple(init.shape) != weight_size:
                        raise ValueError(
                            f"The Tensor specified for weight '{weight_name}' doesn't have the "
                            + "appropiate shape."
                        )
                    return init
            return init(torch.Tensor(*weight_size)) if weight_size else init(torch.Tensor(1))[0]

        for name, size in weight_shapes.items():
            self.qnode_weights[name] = torch.nn.Parameter(
                init_weight(weight_name=name, weight_size=size)
            )

            self.register_parameter(name, self.qnode_weights[name])

    def __str__(self):
        detail = "<Quantum Torch Layer: func={}>"
        return detail.format(self.qnode.func.__name__)

    __repr__ = __str__

    _input_arg = "inputs"
    _initialized = False

    @property
    def input_arg(self):
        """Name of the argument to be used as the input to the Torch layer. Set to ``"inputs"``."""
        return self._input_arg

    @staticmethod
    def set_input_argument(input_name: Text = "inputs") -> None:
        """
        Set the name of the input argument.

        Args:
            input_name (str): Name of the input argument
        """
        TorchLayer._input_arg = input_name