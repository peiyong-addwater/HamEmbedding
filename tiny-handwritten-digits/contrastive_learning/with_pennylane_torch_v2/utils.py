import torch

def cut8x8to4x4PatchesNoPos(img:torch.Tensor):
    batchsize = img.shape[0]
    patches = torch.zeros((batchsize, 4, 16))
    for i in range(2):
        for j in range(2):
            patches[:, 2*i+j, :] = img[:, 4*i:4*i+4, 4*j:4*j+4].flatten(start_dim=1)
    return patches

if __name__ == '__main__':
    img_single = torch.arange(64).reshape(8,8)
    img = torch.stack([img_single, img_single*10, img_single*100])
    print(img)
    patches = cut8x8to4x4PatchesNoPos(img)
    print(patches[0,0,:])
    print(patches)
