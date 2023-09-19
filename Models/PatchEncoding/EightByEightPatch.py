from pennylane.operation import Operation, AnyWires
from .SU4 import SU4, TailLessSU4, HeadlessSU4
import pennylane as qml

def pqc_64(params, wires):
    """
    A 4-qubit, 64-parameter PQC.
    :param params:
    :param wires: 4 qubit
    :return:
    """
    SU4(params[...,:15], wires=[wires[0], wires[1]])
    SU4(params[...,15:30], wires=[wires[2], wires[3]])
    HeadlessSU4(params[...,30:39], wires=[wires[1], wires[2]])
    HeadlessSU4(params[...,39:48], wires=[wires[0], wires[2]])
    HeadlessSU4(params[...,48:57], wires=[wires[1], wires[3]])
    qml.IsingXX(params[...,57], wires=[wires[0], wires[4]])
    qml.IsingYY(params[...,58], wires=[wires[0], wires[4]])
    qml.IsingZZ(params[...,59], wires=[wires[0], wires[4]])
    qml.U2(params[...,60], params[...,61], wires=wires[0])
    qml.U2(params[...,62], params[...,63], wires=wires[4])


