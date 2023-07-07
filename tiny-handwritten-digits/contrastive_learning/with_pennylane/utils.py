import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires

def Reset0(wires:Union[Wires, List[int]]):
    """
    Reset the qubit to |0> with the ResetError channel
    :param wires:
    :return:
    """
    p0 = 1
    p1 = 0
    for wire in wires:
        qml.ResetError(p0=p0, p1=p1, wires=wire)
