import os
import pandas as pd
from torchvision.io import read_image
import torch

class TinyHandwrittenDigitsDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

if __name__ == '__main__':
    import warnings
    from torch.utils.data import DataLoader

    warnings.filterwarnings('ignore')
    from image_transform import DEFAULT_TRANSFORM
    img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
    csv_file="/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/annotated_labels.csv"
    dataset = TinyHandwrittenDigitsDataset(csv_file, img_dir)
    img = dataset[0][0]
    print(img)
    transformed = DEFAULT_TRANSFORM(img)
    print(transformed)

    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=0)
    for batch, (X, y) in enumerate(dataloader):
        print(batch, X.shape, y.shape)
        x = torch.squeeze(X)
        print(x.shape)
        break