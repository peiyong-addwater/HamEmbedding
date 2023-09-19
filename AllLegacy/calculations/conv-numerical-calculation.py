import os.path

import numpy as np
from typing import List, Tuple, Union
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit import Aer
from concurrent.futures import ThreadPoolExecutor
from qiskit.algorithms.optimizers import SPSA
import json
import time
import shutup
import pickle
from qiskit.circuit import ParameterVector


shutup.please()

def su4_circuit(params):
    su4 = QuantumCircuit(2, name='su4')
    su4.u(params[0], params[1], params[2], qubit=0)
    su4.u(params[3], params[4], params[5], qubit=1)
    su4.cx(0,1)
    su4.ry(params[6], 0)
    su4.rz(params[7], 1)
    su4.cx(1, 0)
    su4.ry(params[8], 0)
    su4.cx(0, 1)
    su4.u(params[9], params[10], params[11], 0)
    su4.u(params[12], params[13], params[14], 1)
    su4_inst = su4.to_instruction()
    su4 = QuantumCircuit(2)
    su4.append(su4_inst, list(range(2)))
    return su4

def kernel_5x5(padded_data_in_kernel_view, conv_params):
    """
    45 parameters
    :param padded_data_in_kernel_view:
    :param conv_params:
    :param pooling_params:
    :return:
    """
    qreg = QuantumRegister(4, name="conv")
    creg = ClassicalRegister(4, name='meas')
    circ = QuantumCircuit(qreg, creg, name = "conv-encode-5x5")

    circ.h(qreg)
    # encode the pixel data
    circ.compose(su4_circuit(padded_data_in_kernel_view[:15]), qubits=qreg[:2], inplace=True)
    circ.compose(su4_circuit(padded_data_in_kernel_view[15:]), qubits=qreg[2:], inplace=True)
    # convolution parameters
    circ.compose(su4_circuit(conv_params[:15]), qubits=[qreg[1], qreg[2]], inplace=True)
    circ.compose(su4_circuit(conv_params[15:30]), qubits=[qreg[0], qreg[1]], inplace=True)
    circ.compose(su4_circuit(conv_params[30:45]), qubits=[qreg[2], qreg[3]], inplace=True)
    # measurement
    circ.measure(qreg, creg)



    return circ



if __name__ == '__main__':
    # draw the circuit
    su4_params = ParameterVector("θ", length=30)
    su4_circ = su4_circuit(su4_params)
    su4_circ.decompose().draw(output='mpl', filename='su4.png', style='bw', fold=-1)

    padded_data = ParameterVector("x", length=30)
    kernel_params = ParameterVector("θ", length=45)
    kernel_circ = kernel_5x5(padded_data, kernel_params)
    kernel_circ.draw(output='mpl', filename='conv-5x5-kernel.png', style='bw', fold=-1)

    # random data and kernel parameters
    padded_data = np.random.uniform(low=0., high=2*np.pi, size = 30)
    kernel_params = np.random.uniform(low=0., high=2*np.pi, size = 45)
    kernel_circ = kernel_5x5(padded_data, kernel_params)

    backend_sim = Aer.get_backend('aer_simulator')
    job = backend_sim.run(transpile(kernel_circ, backend_sim), shots = 4096)
    results = job.result()
    counts = results.get_counts()
    print(counts)

