# Based on https://github.com/Qiskit/qiskit-terra/blob/main/qiskit/algorithms/optimizers/adam_amsgrad.py

from typing import Any, Optional, Callable, Dict, Tuple, List
import os

import csv
import numpy as np
from qiskit.utils import algorithm_globals
from qiskit.algorithms.optimizer import OptimizerSupportLevel
from Optimiser import OptimizerSPSAGrad

