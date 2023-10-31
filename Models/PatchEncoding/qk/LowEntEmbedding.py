import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from math import pi

from .su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]

