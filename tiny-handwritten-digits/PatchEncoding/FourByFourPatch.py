from .FourPixelPatch import FourPixelReUpload, FourPixelAmpEnc
from pennylane.operation import Operation, AnyWires
from .SU4 import SU4
import pennylane as qml

class FourByFourNestedReUpload(Operation):
    """
    Encode 16 pixels into 4 qubits;
    The 16 pixels are divided into 4 groups, each group has 4 pixels;
    Each group of 4 pixels is encoded into 2 qubits using 'FourPixelReUpload';
    And have different re-upload parameters for each group of 4 pixels;
    Then only for the 'FourPixelReUpload', the total number of parameters is 6*L1*4
    for a single layer of 'FourByFourPatchReUpload';
    Then the total parameter for four_pixel_encode_parameters should be in shape (...,L2, 6*L1*4)
    Plus a layer of Rot gates and CRot gates, giving us 4*3+3*3=21 parameters per layer of 'FourByFourPatchReUpload';
    Then the shape of sixteen_pixel_parameters should be (...,L2, 21)

    One layer of FourByFourNestedReUpload
    """
