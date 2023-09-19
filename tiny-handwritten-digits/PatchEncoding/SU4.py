import pennylane as qml
from pennylane.operation import Operation, AnyWires

# sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')

class SU4(Operation):
    """
    SU4 gate.
    If the gate is leading gate (15 params), it will start with U3 gates, otherwise (9 params) it will start with the entangling gate.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, weights, wires, leading_gate = True, id=None):
        # interface = qml.math.get_interface(weights)
        shape = qml.math.shape(weights)
        if not (len(shape)==1 or len(shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError("Weights tensor must be 1D or 2D.")
        if shape[-1] != 15 and shape[-1] != 9:
            raise ValueError("Weights tensor must have 15 or 9 elements.")
        if not ((shape[-1] == 15 and leading_gate==True) or (shape[-1] == 9 and leading_gate==False)):
            raise ValueError("Weights tensor must have 15 elements if leading_gate is True, or 9 elements if leading_gate is False.")

        self._hyperparameters = {"leading_gate": leading_gate}

        super().__init__(weights, wires=wires,  id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires, leading_gate):
        op_list = []
        if leading_gate:
            op_list.append(qml.U3(weights[...,0], weights[...,1], weights[...,2], wires=wires[0]))
            op_list.append(qml.U3(weights[...,3], weights[...,4], weights[...,5], wires=wires[1]))
            op_list.append(qml.IsingXX(weights[...,6], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingYY(weights[...,7], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingZZ(weights[...,8], wires=[wires[0], wires[1]]))
            op_list.append(qml.U3(weights[...,9], weights[...,10], weights[...,11], wires=wires[0]))
            op_list.append(qml.U3(weights[...,12], weights[...,13], weights[...,14], wires=wires[1]))
        else:
            op_list.append(qml.IsingXX(weights[...,0], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingYY(weights[...,1], wires=[wires[0], wires[1]]))
            op_list.append(qml.IsingZZ(weights[...,2], wires=[wires[0], wires[1]]))
            op_list.append(qml.U3(weights[...,3], weights[...,4], weights[...,5], wires=wires[0]))
            op_list.append(qml.U3(weights[...,6], weights[...,7], weights[...,8], wires=wires[1]))
        return op_list

class TailLessSU4(Operation):
    """
    SU4 gate, but without the tailing U3 gates.
    """
    num_wires = 2
    grad_method = None

    def __init__(self, weights, wires, id=None):
        # interface = qml.math.get_interface(weights)
        shape = qml.math.shape(weights)
        if not (len(shape)==1 or len(shape)==2): # 2 is when batching, 1 is not batching
            raise ValueError("Weights tensor must be 1D or 2D.")
        if shape[-1] != 9:
            raise ValueError("Weights tensor must have 9 elements.")

        super().__init__(weights, wires=wires,  id=id)

    @property
    def num_params(self):
        return 1

    @staticmethod
    def compute_decomposition(weights, wires):
        op_list = []
        op_list.append(qml.U3(weights[..., 0], weights[..., 1], weights[..., 2], wires=wires[0]))
        op_list.append(qml.U3(weights[..., 3], weights[..., 4], weights[..., 5], wires=wires[1]))
        op_list.append(qml.IsingXX(weights[..., 6], wires=[wires[0], wires[1]]))
        op_list.append(qml.IsingYY(weights[..., 7], wires=[wires[0], wires[1]]))
        op_list.append(qml.IsingZZ(weights[..., 8], wires=[wires[0], wires[1]]))

        return op_list