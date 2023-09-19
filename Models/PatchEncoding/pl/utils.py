import torch

def cut8x8GreyScaleYo4x4PatchesNoPos(img:torch.Tensor):
    """
    Cut 8x8 image to 4x4 patches without position encoding
    Applied to grey scale/binary-valued image
    :param img: shape (batchsize, 8, 8)
    :return: patches: shape (batchsize, 64)
    """
    batchsize = img.shape[0]
    patches = torch.zeros((batchsize, 64))
    for i in range(2):
        for j in range(2):
            patches[:, 16*(2*i+j):16*(2*i+j+1)] = img[:, 4*i:4*i+4, 4*j:4*j+4].flatten(start_dim=1)
    return patches

def cut32x32GreyScaleTo8x8PatchesNoPos(img:torch.Tensor):
    """
    Cut 32x32 image to 8x8 patches without position encoding
    Applied to grey scale/binary-valued image
    :param img: shape (batchsize, 32, 32)
    :return: patches: shape (batchsize, 1024)
    """
    batchsize = img.shape[0]
    patches = torch.zeros((batchsize, 1024))
    for i in range(4):
        for j in range(4):
            patches[:, 64*(4*i+j):64*(4*i+j+1)] = img[:, 8*i:8*i+8, 8*j:8*j+8].flatten(start_dim=1)
