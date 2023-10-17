import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.parametervector import ParameterVectorElement
from qiskit.circuit import Qubit
from typing import Any, Callable, Optional, Sequence, Tuple, List, Union
from .su4 import createTaillessSU4, createHeadlessSU4, createSU4Circ, createRXXRYYRZZCirc
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

def createMemCompCirc(
        num_mem_qubits: int,
        params: QiskitParameter
)->QuantumCircuit:
    """
    A variational circuit with interaction starts from the bottom of the memory qubits to the top ones,
    providing "computation" inside the memory.
    Starting with a layer of U gates: 3*num_mem_qubits parameters
    Then a chain of HeadlessSU4+RxxRyyRzz gates starting from the bottom of the memory qubits: 9*(num_mem_qubits-1)+3*(num_mem_qubits-1) parameters
    Total number of parameters for a single layer: 3*num_mem_qubits + 9*(num_mem_qubits-1) + 3*(num_mem_qubits-1) = 15*num_mem_qubits - 12
    Number of layers is inferred from the length of the parameter vector and number of qubits.
    Args:
        num_mem_qubits: number of the memory qubits
        params: parameters for the variational circuit

    Returns:
        The circuit that implements the computation.
    """
    n_layers = len(params)//(15*num_mem_qubits-12)
    qubit_indices = list(range(num_mem_qubits))
    assert n_layers > 0, "Too few parameters. The number of layers must be positive"
    assert len(params)%(15*num_mem_qubits-12) == 0, f"The number of parameters must be 15*{num_mem_qubits} - 12"

    circ = QuantumCircuit(num_mem_qubits, name="MemComp")
    num_single_layer_params = 15*num_mem_qubits-12
    for i in range(n_layers):
        single_layer_params = params[num_single_layer_params*i:num_single_layer_params*(i+1)]
        u3_params = single_layer_params[:3*num_mem_qubits]
        su4_rxxryyrzz_params = single_layer_params[3*num_mem_qubits:]
        for j in range(num_mem_qubits):
            circ.u(*u3_params[3*j:3*(j+1)], j)
        for j in range(1, num_mem_qubits):
            single_su4_rxxryyrzz_params = su4_rxxryyrzz_params[12*(j-1):12*j]
            circ.append(createHeadlessSU4(single_su4_rxxryyrzz_params[:9]).to_instruction(), [qubit_indices[-j],qubit_indices[-j-1]])
            circ.append(createRXXRYYRZZCirc(single_su4_rxxryyrzz_params[9:]).to_instruction(), [qubit_indices[-j],qubit_indices[-j-1]])
    return circ

def simplePQC(
        num_qubits:int,
        params: QiskitParameter
)->QuantumCircuit:
    """
    A simple parameterised quantum circuit with U gates and controlled-Z gates in a circular fashion.
    Number of layers is inferred from number of qubits and length of parameter vector.
    Args:
        num_qubits: number of qubits
        params: number of parameters

    Returns:
        The circuit that implements the computation.
    """
    n_params = len(params)
    n_layers = n_params//(3*num_qubits)
    assert n_layers > 0, "Too few parameters. The number of layers must be positive"
    assert n_params%(3*num_qubits) == 0, f"The number of parameters must be integer times of 3*{num_qubits}"
    circ = QuantumCircuit(num_qubits, name="SimplePQC-U3CZ")
    for i in range(n_layers):
        layer_params = params[3*num_qubits*i:3*num_qubits*(i+1)]
        for j in range(num_qubits):
            circ.u(*layer_params[3*j:3*(j+1)], j)
        for j in range(num_qubits):
            circ.cz(j, (j+1)%num_qubits)
        circ.barrier()
    return circ

def allInOneAnsatz(
        num_qubits:int,
        params: QiskitParameter
)->QuantumCircuit:
    """
    An all-in-one ansatz for the memory and the patch qubits.
    Following Fig. 6 of https://www.sciencedirect.com/science/article/pii/S089360802300360X
    Starts with a layer of U gates: 3*num_qubits parameters;
    Then a layer of RXXRYYRZZ gates in a circular fashion: 3*num_qubits parameters
    Args:
        num_qubits: number of qubits in the circuit
        params: 6*num_qubits parameters

    Returns:
        The circuit that implements the computation.
    """
    n_params = len(params)
    assert n_params == 6*num_qubits, f"The number of parameters must be 6*{num_qubits}"
    circ = QuantumCircuit(num_qubits, name="AllInOneAnsatz")
    u3_params = params[:3*num_qubits]
    rxxryyrzz_params = params[3*num_qubits:]
    for i in range(num_qubits):
        circ.u(*u3_params[3*i:3*(i+1)], i)
    for i in range(num_qubits):
        circ.append(createRXXRYYRZZCirc(rxxryyrzz_params[3*i:3*(i+1)]).to_instruction(), [i, (i+1)%num_qubits])
    return circ

if __name__ == '__main__':
    n_qubits = 4
    n_mem_qubits = 2
    n_patch_qubits = n_qubits - n_mem_qubits
    param_mem_init = ParameterVector('$\\theta$', 12*n_qubits-9)
    circ_mem_init = createMemStateInitCirc(param_mem_init)
    circ_mem_init.draw('mpl', filename=f'mem_init_{n_qubits}q.png', style='bw')

    param_mem_patch_interact = ParameterVector('$\\theta$', 9*n_mem_qubits*n_patch_qubits)
    circ_mem_patch_interact = createMemPatchInteract(n_mem_qubits, n_patch_qubits, param_mem_patch_interact)
    circ_mem_patch_interact.draw('mpl', filename=f'mem_patch_interact_{n_mem_qubits}-to-{n_patch_qubits}.png', style='bw')

    param_mem_comp = ParameterVector('$\\theta$', 15*n_qubits-12)
    circ_mem_comp = createMemCompCirc(n_qubits, param_mem_comp)
    circ_mem_comp.draw('mpl', filename=f'mem_comp_{n_qubits}q.png', style='bw')

    pqc_params = ParameterVector('$\\theta$', 3*n_qubits*3)
    circ_pqc = simplePQC(n_qubits, pqc_params)
    circ_pqc.draw('mpl', filename=f'pqc_{n_qubits}q_{len(pqc_params)//(3*n_qubits)}_layers.png', style='bw')

    all_in_one_params = ParameterVector('$\\theta$', 6*n_qubits)
    circ_all_in_one = allInOneAnsatz(n_qubits, all_in_one_params)
    circ_all_in_one.draw('mpl', filename=f'all_in_one_{n_qubits}q.png', style='bw')

