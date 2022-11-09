import warnings
warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import IBMQ, Aer, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit import transpile
from qiskit.quantum_info import purity, random_unitary
from qiskit.providers.fake_provider import FakeMontrealV2
from qiskit_experiments.library import StateTomography
import numpy as np

def single_qubit_purity_test(num_u_gates):
    qc = QuantumCircuit(1)
    for i in range(num_u_gates):
        rand_gate = random_unitary(2)
        qc.append(rand_gate, [0])

    qst = StateTomography(qc)
    backend = FakeMontrealV2()
    qstdata = qst.run(backend).block_for_results()
    return qstdata.analysis_results("state_fidelity").value

if __name__ == '__main__':
    for _ in range(1000):
        print(single_qubit_purity_test(20))