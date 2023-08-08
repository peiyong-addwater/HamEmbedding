import pennylane as qml
from pennylane import numpy as pnp
import torch
import numpy as np
from typing import List, Tuple, Union
from pennylane.wires import Wires
import sys
from pennylane.operation import Operation, AnyWires
from reset_gate import ResetZeroState
import math
from four_by_four_patch_encode import FourByFourPatchReUpload

sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

