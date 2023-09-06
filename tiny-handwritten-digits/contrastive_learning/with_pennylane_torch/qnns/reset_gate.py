import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires, Channel



def ResetZeroState(wires:Union[List[int], Wires]):
    for wire in wires:
        _ = qml.measure(wire, reset=True)