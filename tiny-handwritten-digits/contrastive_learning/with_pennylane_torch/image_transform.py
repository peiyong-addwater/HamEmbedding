from torchvision.transforms import v2 as T
import torch

"""
Needed (allowed) image augmentations:
    - ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=12.5, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)
"""

class PixelValueRescale(object):
    """
    Rescale the pixel values to scale_range
    """
    def __init__(self, max_value=255, scaled_max=torch.pi/2):
        self.max_value = max_value
        self.scaled_max = scaled_max

    def __call__(self, image):
        image = image / self.max_value
        image = image * self.scaled_max
        return image

DEFAULT_TRANSFORM = T.Compose(
    [
        T.RandomRotation(12.5, fill=(0,),interpolation=T.InterpolationMode.BILINEAR),
        T.ScaleJitter(target_size=(8,8),scale_range=(0.8,1,2)),
        T.RandomResizedCrop(size=(8,8),scale=((1-0.125)*(1-0.125),1.0)),
        PixelValueRescale(255, torch.pi/2)
    ]
)

#TODO: Rescale the pixel values to [0, Pi/2]