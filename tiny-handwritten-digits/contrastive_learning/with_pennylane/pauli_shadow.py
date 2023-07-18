import pennylane as qml
from pennylane import numpy as pnp
import jax.numpy as jnp
import jax
import numpy as np
from typing import List, Tuple, Union, Optional
from pennylane.wires import Wires
from utils import Reset0
import sys
sys.path.insert(0, '/home/peiyongw/Desktop/Research/QML-ImageClassification')
from backbone_hierarchical_encoding_QFT_Mixing import backboneQFTMixing

UNITARY_ENSEMBLE = {
    0: qml.PauliX,
    1: qml.PauliY,
    2: qml.PauliZ
}
WIRES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
dev = qml.device("default.mixed", wires=WIRES, shots=1)

@qml.qnode(dev, interface="jax")
def single_obs_circuit(
        patched_img: Union[np.ndarray, jnp.ndarray, pnp.ndarray],
        encode_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        single_patch_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        local_patches_phase_parameters: Union[np.ndarray, pnp.ndarray, jnp.ndarray],
        final_layer_parameters:Optional[Union[np.ndarray, jnp.ndarray, pnp.ndarray]] = None,
        final_layer_type: Optional[str] = "generic",
        observables: List = None
):
    """
    The circuit for generating a single snapshot of the Pauli shadow
    :param patched_img: All 16 image patches, a 4 by 4 by 4 array
    :param encode_parameters: "θ", 6N*2*2 element array, where N is the number of "layers" of data-reuploading
    :param single_patch_phase_parameters: "φ", 2*2 element array
    :param local_patches_phase_parameters: "ψ", 2 element array
    :param final_layer_parameters: "η" 12L parameters if using FourQubitParameterisedLayer, 3L parameters if using PermutationInvariantFourQLayer
    :param final_layer_type: "generic" or "permutation-invariant"
    :param observables:
    :return:
    """
    backboneQFTMixing(
        patched_img=patched_img,
        encode_parameters=encode_parameters,
        single_patch_phase_parameters=single_patch_phase_parameters,
        local_patches_phase_parameters=local_patches_phase_parameters,
        final_layer_parameters=final_layer_parameters,
        final_layer_type=final_layer_type,
        wires=WIRES
    )
    return [qml.expval(ob) for ob in observables]



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update("jax_enable_x64", True)

    qml.drawer.use_style("black_white")

    def cut_8x8_to_2x2(img: np.ndarray):
        # img: 8x8 image
        # return: 4x4x4 array, each element in the first 4x4 is a flattened patch
        patches = np.zeros((4, 4, 4))
        for i in range(4):
            for j in range(4):
                patches[i, j] = img[2 * i:2 * i + 2, 2 * j:2 * j + 2].flatten()
        return patches
    img = np.arange(64).reshape(8, 8)
    patches = cut_8x8_to_2x2(img)
    theta = jnp.asarray(np.random.randn(24))
    phi = jnp.asarray(np.random.randn(4))
    psi = jnp.asarray(np.random.randn(2))
    eta = jnp.asarray(np.random.rand(12))
    obs = [qml.PauliZ(0), qml.PauliZ(1), qml.PauliZ(2), qml.PauliZ(3)]
    print(single_obs_circuit(patches, theta, phi, psi, eta, observables=obs))
