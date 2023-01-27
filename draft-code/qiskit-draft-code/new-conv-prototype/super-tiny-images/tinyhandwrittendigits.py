# based on https://github.com/hsjeong5/MNIST-for-Numpy/blob/master/mnist.py
# download data from https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits
import numpy as np
from urllib import request
import gzip
import pickle
from PIL import Image
import pandas as pd

filename = [
    "optdigits.tes",
    "optdigits.tra",
    "optdigits.names"
]


def download_tiny_handwritten_digits():
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
    for name in filename:
        print("Downloading "+name+"...")
        request.urlretrieve(base_url+name, name)
    print("Download complete.")

def save_mnist():
    mnist = {}
    mnist["training_images"] = []
    mnist["training_labels"] = []
    mnist["test_images"], mnist["test_labels"] = [], []
    train_filename = "optdigits.tra"
    test_filename = "optdigits.tes"
    train_df = pd.read_csv(train_filename)
    train_array = train_df.to_numpy()
    test_df = pd.read_csv(test_filename)
    test_array = test_df.to_numpy()
    for c in train_array:
        image = c[:-1]/16
        label = c[-1]
        mnist["training_images"].append(image.reshape(8, 8))
        mnist["training_labels"].append(label)
    for c in test_array:
        image = c[:-1]/16
        label = c[-1]
        mnist["test_images"].append(image.reshape(8, 8))
        mnist["test_labels"].append(label)

    with open("tiny-handwritten.pkl", 'wb') as f:
        pickle.dump(mnist,f)
    print("Save complete.")

def init():
    download_tiny_handwritten_digits()
    save_mnist()

def load():
    with open("tiny-handwritten.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]





if __name__ == '__main__':
    init()
    data = load()
    train_images = data[0]
    train_labels = data[1]
    idx = 1123
    example_image_array = train_images[idx]
    example_image_formatted = (example_image_array * 255 / np.max(example_image_array)).astype('uint8')
    example_image = Image.fromarray(example_image_formatted)
    example_image.save(f"example_tiny_handwritten_digit_{train_labels[idx]}.png")