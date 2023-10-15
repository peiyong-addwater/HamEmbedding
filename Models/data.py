import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn import datasets


def cut8x8GreyScaleYo4x4PatchesNoPos(img:torch.Tensor):
    """
    Cut 8x8 image to 4x4 patches without position encoding
    Applied to grey scale/binary-valued image
    :param img: shape ( 8, 8)
    :return: patches: shape ( 64)
    """
    patches = torch.zeros(64)
    for i in range(2):
        for j in range(2):
            patches[16*(2*i+j):16*(2*i+j+1)] = img[4*i:4*i+4, 4*j:4*j+4].reshape(-1)
    return patches

class PatchedDigitsDataset(Dataset):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.digits = datasets.load_digits()
        self.images = (self.digits.images/16)*(2*torch.pi)
        self.lables = self.digits.target

    def __len__(self):
        return len(self.digits.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = torch.tensor(self.images[idx])
        label = torch.tensor(self.lables[idx])
        #print(image.shape)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        image = torch.squeeze(image)
        patches = cut8x8GreyScaleYo4x4PatchesNoPos(image)
        return patches, label

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = PatchedDigitsDataset()
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    for batch, (X, y) in enumerate(dataloader):
        print(batch, X.shape, y.shape)
        print(X)
        print(y)
        break