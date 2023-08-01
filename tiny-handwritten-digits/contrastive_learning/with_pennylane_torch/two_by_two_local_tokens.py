import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
from reset_gate import Reset0
from token_mixing_layers import FourQubitGenericParameterisedLayer, su4_gate
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

