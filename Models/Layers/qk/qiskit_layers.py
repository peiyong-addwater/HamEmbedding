import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import Qubit
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ
from math import pi

QiskitParameter = Union[ParameterVector, List[Parameter], List[ParameterVectorElement]]
QiskitQubits = Union[List[int], List[Qubit], QuantumRegister]

def createMemStateInitCirc(
        params: QiskitParameter
)->QuantumCircuit:
    """
    Initial state of the memory qubits. Trainable.
    Start with a layer of H gates,
    then a layer of U3 gates,
    then a brickwall of Headless SU4 gates.
    Total number of parameters: 3 * n_qubits + 9 * (n_qubits-1) = 12 * n_qubits - 9
    The number of qubits is determined by the length of the parameter vector.
    Args:
        params: number of elements = 3 * n_qubits + 9 * (n_qubits-1)= 12 * n_qubits - 9

    Returns:
        The circuit that initialises the memory qubits.
    """
    n_qubits = (len(params)+9)//12
    assert (len(params)+9)%12 == 0, f"The number of parameters must be 12 * {n_qubits} - 9"
    assert n_qubits > 0, "Too few parameters. The number of qubits must be positive"
    assert (len(params)+9)/12 == n_qubits, "The number of parameters must be 12 * n_qubits - 9"
    assert n_qubits>=2, "The number of qubits must be at least 2"

    circ = QuantumCircuit(n_qubits, name="InitialiseMemState")
    for i in range(n_qubits):
        circ.h(i)
        circ.u(*params[3*i:3*(i+1)], i)
    if n_qubits == 2:
        circ.append(createHeadlessSU4(params[3*n_qubits:]).to_instruction(), [0,1])
    else:
        if n_qubits%2 == 0:
            headless_su4_count = 0
            for i in range(n_qubits//2):
                circ.append(createHeadlessSU4(params[3*n_qubits+9*headless_su4_count:3*n_qubits+9*(headless_su4_count+1)]).to_instruction(), [2*i,2*i+1])
                headless_su4_count += 1
            for i in range(n_qubits//2-1):
                circ.append(createHeadlessSU4(params[3*n_qubits+9*headless_su4_count:3*n_qubits+9*(headless_su4_count+1)]).to_instruction(), [2*i+1,2*i+2])
                headless_su4_count += 1
        else:
            headless_su4_count = 0
            for i in range((n_qubits-1)//2):
                circ.append(createHeadlessSU4(params[3*n_qubits+9*headless_su4_count:3*n_qubits+9*(headless_su4_count+1)]).to_instruction(), [2*i,2*i+1])
                headless_su4_count += 1
            for i in range((n_qubits-1)//2):
                circ.append(createHeadlessSU4(params[3*n_qubits+9*headless_su4_count:3*n_qubits+9*(headless_su4_count+1)]).to_instruction(), [2*i+1,2*i+2])
                headless_su4_count += 1

    return circ

def createMemPatchInteract(
        num_mem_qubits: int,
        num_patch_qubits: int,
        params: QiskitParameter
)->QuantumCircuit:
    """
    Provides TaillessSU4-based interaction between the last num_mem_qubits and the first num_patch_qubits.
    Args:
        num_mem_qubits: number of memory qubits involved in the interaction
        num_patch_qubits: number of patch qubits involved in the interaction
        params: 9*(num_mem_qubits*num_patch_qubits)

    Returns:
        The circuit that implements the interaction.
    """
    total_num_qubits = num_mem_qubits + num_patch_qubits
    assert len(params) == 9*num_mem_qubits*num_patch_qubits, f"The number of parameters must be 9*{num_mem_qubits}*{num_patch_qubits}"
    assert num_mem_qubits > 0, "The number of memory qubits must be positive"
    assert num_patch_qubits > 0, "The number of patch qubits must be positive"
    assert total_num_qubits > 1, "The total number of qubits must be at least 2"

    qubit_indices = list(range(total_num_qubits))
    mem_qubits = qubit_indices[:num_mem_qubits]
    patch_qubits = qubit_indices[num_mem_qubits:]

    circ = QuantumCircuit(total_num_qubits, name="MemPatchInteract")
    tailless_su4_count = 0
    for i in range(num_mem_qubits):
        for j in range(num_patch_qubits):
            circ.append(createTaillessSU4(params[9*tailless_su4_count:9*(tailless_su4_count+1)]).to_instruction(), [mem_qubits[i], patch_qubits[j]])
            tailless_su4_count += 1
    return circ


if __name__ == '__main__':
    n_qubits = 5
    n_mem_qubits = 2
    n_patch_qubits = n_qubits - n_mem_qubits
    param_mem_init = ParameterVector('$\\theta$', 12*n_qubits-9)
    circ_mem_init = createMemStateInitCirc(param_mem_init)
    circ_mem_init.draw('mpl', filename=f'mem_init_{n_qubits}q.png', style='bw')

    param_mem_patch_interact = ParameterVector('$\\theta$', 9*n_mem_qubits*n_patch_qubits)
    circ_mem_patch_interact = createMemPatchInteract(n_mem_qubits, n_patch_qubits, param_mem_patch_interact)
    circ_mem_patch_interact.draw('mpl', filename=f'mem_patch_interact_{n_mem_qubits}-to-{n_patch_qubits}.png', style='bw')

