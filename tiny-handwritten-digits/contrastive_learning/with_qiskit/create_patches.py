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

if __name__ == '__main__':

    img = np.arange(64).reshape(8,8)
    print(img)
    patches = cut_8x8_to_2x2(img)
    print(patches[0,0,:])
    print(patches)

    with open(os.path.join(data_dir, "tiny-handwritten-as-rotation-angles.pkl"), 'rb') as f:
        mnist = pickle.load(f)
        train_imgs, train_labels, test_imgs, test_labels = mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

    patched_img_data = dict()

    patched_img_data["training_patches"] = [cut_8x8_to_2x2(img) for img in train_imgs]
    patched_img_data["training_labels"] = train_labels
    patched_img_data["test_patches"] = [cut_8x8_to_2x2(img) for img in test_imgs]
    patched_img_data["test_labels"] = test_labels

    with open(os.path.join(cwd, "tiny-handwritten-as-rotation-angles-patches.pkl"), 'wb') as f:
        pickle.dump(patched_img_data, f)