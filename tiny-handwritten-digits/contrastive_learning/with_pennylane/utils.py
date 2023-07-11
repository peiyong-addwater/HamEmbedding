import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import pennylane.math as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
from pennylane.operation import AnyWires, Channel

def Reset0(wires:Union[Wires, List[int]]):
    """
    Reset the qubit to |0> with the ResetError channel
    :param wires:
    :return:
    """
    p0 = 1
    p1 = 0
    for wire in wires:
        Reset(wires=wire, id="|0>")


class Reset(Channel):
    r"""
    A reset channel, modified from the original pennylane ResetError channel to allow for resetting to |0> or |1> with different probabilities.
    ResetError channel see https://docs.pennylane.ai/en/stable/code/api/pennylane.ResetError.html
    We just set p0 = 1 and p1 = 0 to reset to |0> with probability 1.

    **Details:**

    * Number of wires: 1
    * Number of parameters: 0

    Args:
        wires (Sequence[int] or int): the wire the channel acts on
        do_queue (bool): Indicates whether the operator should be
            immediately pushed into the Operator queue (optional).
            This argument is deprecated, instead of setting it to ``False``
            use :meth:`~.queuing.QueuingManager.stop_recording`.
        id (str or None): String representing the operation (optional)
    """
    num_params = 0
    num_wires = 1
    grad_method = None

    def __init__(self,  wires, do_queue=None, id=None):
        super().__init__( wires=wires, do_queue=do_queue, id=id)

    @staticmethod
    def compute_kraus_matrices():  # pylint:disable=arguments-differ
        p_0 = 1
        p_1 = 0
        if not np.is_abstract(p_0) and not 0.0 <= p_0 <= 1.0:
            raise ValueError("p_0 must be in the interval [0,1]")

        if not np.is_abstract(p_1) and not 0.0 <= p_1 <= 1.0:
            raise ValueError("p_1 must be in the interval [0,1]")

        if not np.is_abstract(p_0 + p_1) and not 0.0 <= p_0 + p_1 <= 1.0:
            raise ValueError("p_0 + p_1 must be in the interval [0,1]")

        interface = np.get_interface(p_0, p_1)
        p_0, p_1 = np.coerce([p_0, p_1], like=interface)
        K0 = np.sqrt(1 - p_0 - p_1 + np.eps) * np.convert_like(np.cast_like(np.eye(2), p_0), p_0)
        K1 = np.sqrt(p_0 + np.eps) * np.convert_like(
            np.cast_like(np.array([[1, 0], [0, 0]]), p_0), p_0
        )
        K2 = np.sqrt(p_0 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 1], [0, 0]]), p_0), p_0
        )
        K3 = np.sqrt(p_1 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 0], [1, 0]]), p_0), p_0
        )
        K4 = np.sqrt(p_1 + np.eps) * np.convert_like(
            np.cast_like(np.array([[0, 0], [0, 1]]), p_0), p_0
        )

        return [K0, K1, K2, K3, K4]
