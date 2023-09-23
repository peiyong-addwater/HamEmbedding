# From https://github.com/ryanlevy/shadow-tutorial/blob/main/Tutorial_Shadow_State_Tomography.ipynb
import numpy as np
import matplotlib.pyplot as plt
import qiskit

pauli_list = [
    np.eye(2),
    np.array([[0.0, 1.0], [1.0, 0.0]]),
    np.array([[0, -1.0j], [1.0j, 0.0]]),
    np.array([[1.0, 0.0], [0.0, -1.0]]),
]
s_to_pauli = {
    "I": pauli_list[0],
    "X": pauli_list[1],
    "Y": pauli_list[2],
    "Z": pauli_list[3],
}

# The state we want to reconstruct
def channel(N, qc):
    '''create an N qubit GHZ state '''
    qc.h(0)
    if N >= 2: qc.cx(0, 1)
    if N >= 3: qc.cx(0, 2)
    if N >= 4: qc.cx(1, 3)
    if N > 4: raise NotImplementedError(f"{N} not implemented!")


def bitGateMap(qc, g, qi):
    '''Map X/Y/Z string to qiskit ops'''
    if g == "X":
        qc.h(qi)
    elif g == "Y":
        qc.sdg(qi)
        qc.h(qi)
    elif g == "Z":
        pass
    else:
        raise NotImplementedError(f"Unknown gate {g}")


def Minv(N, X):
    '''inverse shadow channel'''
    return ((2 ** N + 1.)) * X - np.eye(2 ** N)

nShadows = 2048
reps = 1
N = 4
rng = np.random.default_rng(1717)
cliffords = [qiskit.quantum_info.random_clifford(N, seed=rng) for _ in range(nShadows)]

qc = qiskit.QuantumCircuit(N)
channel(N,qc)

results = []
for cliff in cliffords:
    qc_c  = qc.compose(cliff.to_circuit())
    counts = qiskit.quantum_info.Statevector(qc_c).sample_counts(reps)
    results.append(counts)

print(qc)
"""
     ┌───┐               
q_0: ┤ H ├──■────■───────
     └───┘┌─┴─┐  │       
q_1: ─────┤ X ├──┼────■──
          └───┘┌─┴─┐  │  
q_2: ──────────┤ X ├──┼──
               └───┘┌─┴─┐
q_3: ───────────────┤ X ├
                    └───┘
"""
print(qc_c)
"""
     ┌───┐          ┌───┐          ┌───┐          ┌───┐┌───┐                  »
q_0: ┤ H ├──■────■──┤ S ├───────■──┤ X ├──■───────┤ X ├┤ X ├──────────────────»
     └───┘┌─┴─┐  │  └───┘┌───┐  │  └─┬─┘┌─┴─┐┌───┐└─┬─┘├───┤┌───┐             »
q_1: ─────┤ X ├──┼────■──┤ H ├──┼────┼──┤ X ├┤ H ├──■──┤ H ├┤ S ├─X────────■──»
          └───┘┌─┴─┐  │  └───┘  │    │  ├───┤├───┤     └───┘└───┘ │      ┌─┴─┐»
q_2: ──────────┤ X ├──┼─────────┼────■──┤ S ├┤ H ├────────────────X───■──┤ X ├»
               └───┘┌─┴─┐┌───┐┌─┴─┐┌───┐└───┘└───┘                  ┌─┴─┐├───┤»
q_3: ───────────────┤ X ├┤ H ├┤ X ├┤ S ├────────────────────────────┤ X ├┤ H ├»
                    └───┘└───┘└───┘└───┘                            └───┘└───┘»
«                    
«q_0: ───────────────
«          ┌───┐     
«q_1: ──■──┤ X ├─────
«       │  ├───┤     
«q_2: ──┼──┤ Z ├─────
«     ┌─┴─┐├───┤┌───┐
«q_3: ┤ X ├┤ H ├┤ Z ├
«     └───┘└───┘└───┘
"""


# Clifford Shadows
shadows = []
for cliff, res in zip(cliffords, results):
    mat    = cliff.adjoint().to_matrix()
    for bit,count in res.items():
        Ub = mat[:,int(bit,2)] # this is Udag|b>
        shadows.append(Minv(N,np.outer(Ub,Ub.conj()))*count)

rho_shadow = np.sum(shadows,axis=0)/(nShadows*reps)

rho_actual = qiskit.quantum_info.DensityMatrix(qc).data


plt.subplot(121)
plt.suptitle("Correct")
plt.imshow(rho_actual.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_actual.imag,vmax=0.7,vmin=-0.7)
plt.savefig("correct.png")


plt.subplot(121)
plt.suptitle(f"Shadow(Clifford)-{nShadows}-shadows")
plt.imshow(rho_shadow.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_shadow.imag,vmax=0.7,vmin=-0.7)
plt.savefig(f"shadow-clifford-{nShadows}-shadows.png")

qiskit.visualization.state_visualization.plot_state_city(rho_actual,title="Correct").savefig("correct-city.png")
qiskit.visualization.state_visualization.plot_state_city(rho_shadow,title="Shadow (clifford)").savefig(f"shadow-clifford-{nShadows}-shadows-city.png")
plt.close('all')

# Pauli
nShadows = 2048
N = 4

rng = np.random.default_rng(1717)
scheme = [rng.choice(['X', 'Y', 'Z'], size=N) for _ in range(nShadows)]
labels, counts = np.unique(scheme, axis=0, return_counts=True)

qc = qiskit.QuantumCircuit(N)
channel(N, qc)

results = []
for bit_string, count in zip(labels, counts):
    qc_m = qc.copy()
    # rotate the basis for each qubit
    for i, bit in enumerate(bit_string): bitGateMap(qc_m, bit, i)
    counts = qiskit.quantum_info.Statevector(qc_m).sample_counts(count)
    results.append(counts)

# Note: Qiskit uses little-endian bit ordering
def rotGate(g):
    '''produces gate U such that U|psi> is in Pauli basis g'''
    if g=="X":
        return 1/np.sqrt(2)*np.array([[1.,1.],[1.,-1.]])
    elif g=="Y":
        return 1/np.sqrt(2)*np.array([[1.,-1.0j],[1.,1.j]])
    elif g=="Z":
        return np.eye(2)
    else:
        raise NotImplementedError(f"Unknown gate {g}")

shadows = []
shots = 0
for pauli_string,counts in zip(labels,results):
    # iterate over measurements
     for bit,count in counts.items():
        mat = 1.
        for i,bi in enumerate(bit[::-1]):
            b = rotGate(pauli_string[i])[int(bi),:]
            mat = np.kron(Minv(1,np.outer(b.conj(),b)),mat)
        shadows.append(mat*count)
        shots+=count

rho_shadow = np.sum(shadows,axis=0)/(shots)

rho_actual = qiskit.quantum_info.DensityMatrix(qc).data


plt.subplot(121)
plt.suptitle("Correct-Pauli")
plt.imshow(rho_actual.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_actual.imag,vmax=0.7,vmin=-0.7)
plt.savefig("correct-pauli.png")


plt.subplot(121)
plt.suptitle(f"Shadow(Pauli)-{nShadows}-shadows")
plt.imshow(rho_shadow.real,vmax=0.7,vmin=-0.7)
plt.subplot(122)
plt.imshow(rho_shadow.imag,vmax=0.7,vmin=-0.7)
plt.savefig(f"shadow-pauli-{nShadows}-shadows.png")

qiskit.visualization.state_visualization.plot_state_city(rho_actual,title="Correct").savefig("correct-city-pauli.png")
qiskit.visualization.state_visualization.plot_state_city(rho_shadow,title=f"Shadow (Pauli) {nShadows} shadows").savefig(f"shadow-pauli-{nShadows}-shadows-city.png")



