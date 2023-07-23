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
    from tqdm import tqdm

    with open(os.path.join(data_dir, "augmentation_arrays.pickle"), 'rb') as f:
        # all 0 to 255
        mnist = pickle.load(f)
        train, test = mnist["train"], mnist["test"]

    print(len(train))
    print(len(test))
    print(train[0]["original"].shape)

    patched_img_data = dict()
    patched_img_data["train"] = []
    patched_img_data["test"] = []
    for c in tqdm(train):
        patched_img_data["train"].append({
            "original": cut_8x8_to_2x2(c["original"]*(2*np.pi)/255),
            "label": c["label"], # 0~9
            "augmentations": [cut_8x8_to_2x2(a*(2*np.pi)/255) for a in c["augmentations"]]
        })
    for c in tqdm(test):
        patched_img_data["test"].append({
            "original": cut_8x8_to_2x2(c["original"]*(2*np.pi)/255),
            "label": c["label"],  # 0~9
            "augmentations": [cut_8x8_to_2x2(a*(2*np.pi)/255) for a in c["augmentations"]]
        })


    print(patched_img_data["train"][0]["original"])
    print(patched_img_data["train"][0]["original"].shape)
    print(patched_img_data["train"][0])

    with open(os.path.join(data_dir, "tiny-handwritten-with-augmented-as-rotation-angles-patches.pkl"), 'wb') as f:
        pickle.dump(patched_img_data, f)