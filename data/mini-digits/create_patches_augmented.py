import numpy as np
import os
import pickle


# the pixels in the input image should already be converted to angles of rotation, i.e. 0~2pi
# create 2x2 patches for the 8x8 image, and flatten the image patches into a 1-d array

data_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits"
cwd = os.getcwd()

def cut_8x8_to_2x2(img:np.ndarray):
    # img: 8x8 image
    # return: 4x4x4 array, each element in the first 4x4 is a flattend patch
    patches = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            patches[i,j] = img[2*i:2*i+2, 2*j:2*j+2].flatten()
    return patches