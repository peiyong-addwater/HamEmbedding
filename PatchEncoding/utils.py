import torch

def cut8x8GreyScaleto4x4PatchesNoPos(img:torch.Tensor):
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