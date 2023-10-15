import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from qiskit.circuit.parametervector import ParameterVectorElement

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]

def createSU4Circ(params:QiskitParameter)->QuantumCircuit:
    """
    Create the full SU4 circuit with the given parameters.
    Args:
        params: parameters of the SU4 circuit. The length of the parameters must be 15.

    Returns:
        QuantumCircuit: the SU4 circuit.
    """
    su4 = QuantumCircuit(2, name='SU4')
    su4.u(params[0], params[1], params[2], 0)
    su4.u(params[3], params[4], params[5], 1)
    su4.rxx(params[6], 0, 1)
    su4.ryy(params[7], 0, 1)
    su4.rzz(params[8], 0, 1)
    su4.u(params[9], params[10], params[11], 0)
    su4.u(params[12], params[13], params[14], 1)
    return su4

def createHeadlessSU4(params:QiskitParameter)->QuantumCircuit:
    """
    Create the headless SU4 circuit with the given parameters.
    HeadlessSU4 is the SU4 circuit without the leading U3 gates.
    Args:
        params: parameters of the SU4 circuit. The length of the parameters must be 9.

    Returns:
        QuantumCircuit: the headless SU4 circuit.
    """
    su4 = QuantumCircuit(2, name='HeadlessSU4')
    su4.rxx(params[0], 0, 1)
    su4.ryy(params[1], 0, 1)
    su4.rzz(params[2], 0, 1)
    su4.u(params[3], params[4], params[5], 0)
    su4.u(params[6], params[7], params[8], 1)
    return su4

def createTaillessSU4(params:QiskitParameter)->QuantumCircuit:
    """
    Create the tailless SU4 circuit with the given parameters.
    TaillessSU4 is the SU4 circuit without the trailing U3 gates.
    Args:
        params: parameters of the SU4 circuit. The length of the parameters must be 9.

    Returns:
        QuantumCircuit: the tailless SU4 circuit.
    """
    su4 = QuantumCircuit(2, name='TaillessSU4')
    su4.u(params[0], params[1], params[2], 0)
    su4.u(params[3], params[4], params[5], 1)
    su4.rxx(params[6], 0, 1)
    su4.ryy(params[7], 0, 1)
    su4.rzz(params[8], 0, 1)
    return su4

def createRXXRYYRZZCirc(params:QiskitParameter)->QuantumCircuit:
    """
    Create the RXXRYYRZZ circuit with the given parameters.
    Args:
        params: parameters of the RXXRYYRZZ circuit. The length of the parameters must be 3.

    Returns:
        QuantumCircuit: the RXXRYYRZZ circuit.
    """
    circ = QuantumCircuit(2, name='RXXRYYRZZ')
    circ.rxx(params[0], 0, 1)
    circ.ryy(params[1], 0, 1)
    circ.rzz(params[2], 0, 1)
    return circ

