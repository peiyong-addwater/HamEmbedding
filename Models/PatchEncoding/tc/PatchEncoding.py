from functools import wraps
from typing import Any, Callable, Optional, Sequence, Tuple

import numpy as np

from su4 import applySU4Gate, applyHeadlessSU4, applyTaillessSU4

import tensorcircuit as tc

Circuit = tc.Circuit
Tensor = Any
Graph = Any

def applyFourPixelReUploadCirc(
        circ:Circuit,
        pixels:Tensor,
        encode_params:Tensor,
        reps:int,
        wires:Sequence[int],
)->Circuit:
    """
    Encode 4 pixels with a 2-qubit data re-uploading circuit.
    Args:
        circ: the tensorcircuit.circuit.Circuit object this function is being applied to
        pixels: the input pixel data. Four element array.
        encode_params: parameters for data re-uploading
        reps: number of repetitions for data re-uploading
        wires: the qubits to be used. Number of qubits: 2

    Returns:
        the updated circuit with the data re-uploading layers applied
    """
    for i in range(reps):
        circ.rx(wires[0], theta = pixels[..., 0])
        circ.ry(wires[0], theta = pixels[..., 1])
        circ.rx(wires[1], theta = pixels[..., 2])
        circ.ry(wires[1], theta = pixels[..., 3])
        circ.cz(wires[0], wires[1])
        circ.U(wires[0], theta = encode_params[..., 6 * i], phi = encode_params[..., 6 * i + 1], lbd = encode_params[..., 6 * i + 2])
        circ.U(wires[1], theta = encode_params[..., 6 * i + 3], phi = encode_params[..., 6 * i + 4], lbd = encode_params[..., 6 * i + 5])
        circ.cz(wires[0], wires[1])
        circ.barrier_instruction(wires)
    return circ




if __name__ == '__main__':
    # test
    import jax
    import jax.numpy as jnp

    K = tc.set_backend("jax")
    circ = tc.Circuit(2)
    pixels = jnp.array([1, 2, 3, 4])
    encode_params = jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    circ = applyFourPixelReUploadCirc(circ, pixels, encode_params, 2, [0, 1])
    print(circ.to_qiskit())
    print(circ.state())


    # def loss(params, n):
    #     c = tc.Circuit(n)
    #     for i in range(n):
    #         c.rx(i, theta=params[0, i])
    #     for i in range(n):
    #         c.rz(i, theta=params[1, i])
    #     loss = 0.0
    #     for i in range(n):
    #         loss += c.expectation([tc.gates.z(), [i]])
    #     return K.real(loss)
    #
    #
    # n = 10
    # vgf = K.jit(K.value_and_grad(loss), static_argnums=1)
    # params = K.implicit_randn([2, n])
    # print(vgf(params, n))  # get the quantum loss and the gradient
