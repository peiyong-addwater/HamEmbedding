import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ
from math import pi

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]

def createMemStateInitCirc(
        params: QiskitParameter,
        n_qubits: int
)->QuantumCircuit:
    """
    Initial state of the memory qubits. Trainable.
    Start with a layer of H gates,
    then a layer of U3 gates,
    then a brickwall of Headless SU4 gates.
    Total number of parameters: 3 * n_qubits + 9 * (n_qubits-1)
    Args:
        params:
        n_qubits:

    Returns:
        The circuit that initialises the memory qubits.
    """