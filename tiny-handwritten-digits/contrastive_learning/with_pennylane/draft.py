import pennylane as qml
import numpy as np
import torch
from torch.autograd import Variable
np.random.seed(42)

# we generate a three-dimensional random vector by sampling
# each entry from a standard normal distribution
v = np.random.normal(0, 1, 3)

# purity of the target state
purity = 0.66

# create a random Bloch vector with the specified purity
bloch_v = Variable(
    torch.tensor(np.sqrt(2 * purity - 1) * v / np.sqrt(np.sum(v ** 2))),
    requires_grad=False
)

# array of Pauli matrices (will be useful later)
Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False).to(device=torch.device('cuda'))
Paulis[0] = torch.tensor([[0, 1], [1, 0]]).to(device=torch.device('cuda'))
Paulis[1] = torch.tensor([[0, -1j], [1j, 0]]).to(device=torch.device('cuda'))
Paulis[2] = torch.tensor([[1, 0], [0, -1]]).to(device=torch.device('cuda'))

# number of qubits in the circuit
nr_qubits = 3
# number of layers in the circuit
nr_layers = 2

# randomly initialize parameters from a normal distribution
params = np.random.normal(0, np.pi, (nr_qubits, nr_layers, 3))
params = Variable(torch.tensor(params), requires_grad=True)

# a layer of the circuit ansatz
def layer(params, j):
    for i in range(nr_qubits):
        qml.RX(params[i, j, 0], wires=i)
        qml.RY(params[i, j, 1], wires=i)
        qml.RZ(params[i, j, 2], wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[0, 2])
    qml.CNOT(wires=[1, 2])


dev = qml.device("default.mixed", wires=3)
@qml.qnode(dev, interface="torch")
def circuit(params, A):

    # repeatedly apply each layer in the circuit
    for j in range(nr_layers):
        layer(params, j)

    # returns the expectation of the input matrix A on the first qubit
    return qml.expval(qml.Hermitian(A, wires=0))

# cost function
def cost_fn(params):
    cost = 0
    for k in range(3):
        cost += torch.abs(circuit(params, Paulis[k]) - bloch_v[k])

    return cost


# set up the optimizer
opt = torch.optim.Adam([params], lr=0.1)

# number of steps in the optimization routine
steps = 200

# the final stage of optimization isn't always the best, so we keep track of
# the best parameters along the way
best_cost = cost_fn(params)
best_params = np.zeros((nr_qubits, nr_layers, 3))
params = params.to(device=torch.device('cuda'))
print("Cost after 0 steps is {:.4f}".format(cost_fn(params)))

# optimization begins
for n in range(steps):
    opt.zero_grad()
    loss = cost_fn(params)
    loss.backward()
    opt.step()

    # keeps track of best parameters
    if loss < best_cost:
        best_cost = loss
        best_params = params

    # Keep track of progress every 10 steps
    if n % 10 == 9 or n == steps - 1:
        print("Cost after {} steps is {:.4f}".format(n + 1, loss))

# calculate the Bloch vector of the output state
output_bloch_v = np.zeros(3)
for l in range(3):
    output_bloch_v[l] = circuit(best_params, Paulis[l])

# print results
print("Target Bloch vector = ", bloch_v.numpy())
print("Output Bloch vector = ", output_bloch_v)