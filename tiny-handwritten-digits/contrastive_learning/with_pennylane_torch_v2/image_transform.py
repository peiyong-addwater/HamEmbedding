from torchvision.transforms import v2 as T

"""
Needed (allowed) image augmentations:
    - ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=12.5, p=1, border_mode=cv2.BORDER_CONSTANT, value=0)
"""

DEFAULT_TRANSFORM = T.Compose(
    [
        T.RandomRotation(12.5, fill=(0,),interpolation=T.InterpolationMode.BILINEAR),
        T.ScaleJitter(target_size=(8,8),scale_range=(0.8,1,2)),
        T.RandomResizedCrop(size=(8,8),scale=((1-0.125)*(1-0.125),1.0))
    ]
)

#TODO: Rescale the pixel values to [0, Pi/2]