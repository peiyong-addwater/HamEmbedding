from qiskit.providers.aer import AerError
from qiskit import QuantumCircuit
from qiskit import Aer, transpile
from qiskit.tools.visualization import plot_histogram, plot_state_city
import qiskit.quantum_info as qi

print(Aer.backends())
try:
    simulator_gpu = Aer.get_backend('aer_simulator')
    simulator_gpu.set_options(device='GPU')
    print("Yay, GPU!")
except AerError as e:
    print(e)

from qiskit_ibm_provider import IBMProvider
#IBMProvider.save_account(token='cd8958c6209f1fc8c60a933cf4dc853f3ff7c170b3f98389930dce81de4d752e876577789c085d27e8c40a4a03b80dc2e4e19fd2a310353b90514cd0c7ccf9d5')
provider = IBMProvider()
print(provider.backends())
