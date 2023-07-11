import random
import re
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import json
import albumentations as A
import os
import pickle
import numpy as np

image_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
example_save_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/example_augmentation"
augmentation_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/augmented_images"
labels = [0,1,2,3,4,5,6,7,8,9]

def cut_8x8_to_2x2(img:np.ndarray):
    # img: 8x8 image
    # return: 4x4x4 array, each element in the first 4x4 is a flattend patch
    patches = np.zeros((4,4,4))
    for i in range(4):
        for j in range(4):
            patches[i,j] = img[2*i:2*i+2, 2*j:2*j+2].flatten()
    return patches

# list image filenames
img_files = os.listdir(image_folder)
random.shuffle(img_files)

filename_dict = {}
filename_dict["train"] = []
filename_dict["test"] = []
img_dict = {}
img_dict["train"] = []
img_dict["test"] = []

transform = A.Compose([
    #A.CLAHE(),
    #A.RandomRotate90(),
    #A.Transpose(),
    A.ShiftScaleRotate(shift_limit=0.125, scale_limit=0.2, rotate_limit=12.5, p=1, border_mode=cv2.BORDER_CONSTANT, value=0),
    #A.Blur(blur_limit=3),
    #A.OpticalDistortion(),
    #A.GridDistortion(),
    #A.HueSaturationValue(),
])

"""
# show some example augmented images
for label in labels:
    for filename in img_files:
        if f"_label_{label}.png" in filename:
            img = cv2.imread(os.path.join(image_folder, filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.imwrite(os.path.join(example_save_folder, f"example_image_label_{label}.png"), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


            for i in range(5):
                transformed = transform(image=img)
                transformed_image = transformed["image"]


                save_name = f"example_image_label_{label}_aug_{i}.png"
                cv2.imwrite(os.path.join(example_save_folder, save_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY))
"""

# augment all images
NUM_AUGMENTATIONS = 10
for filename in tqdm(img_files):
    img = cv2.imread(os.path.join(image_folder, filename))
    cv2.imwrite(os.path.join(image_folder, filename), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img_array = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    str_contain_label = filename.split("_")[-1]
    label = str_contain_label.split(".")[0]
    single_sample_filename_dict = {}
    single_sample_filename_dict["label"] = label
    single_sample_filename_dict["original"] = os.path.join(image_folder, filename)
    single_sample_array_dict = {}
    single_sample_array_dict["label"] = label
    single_sample_array_dict["original"] = img_array
    augs_filename = []
    augs_array = []
    for i in range(NUM_AUGMENTATIONS):
        transformed = transform(image=img)
        transformed_image = transformed["image"]
        transformed_image_array = np.array(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY))
        augs_array.append(transformed_image_array)
        save_name = f"{filename.split('.')[0]}_aug_{i}.png"
        augs_filename.append(os.path.join(augmentation_folder, save_name))
        cv2.imwrite(os.path.join(augmentation_folder, save_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY))
    single_sample_filename_dict["augmentations"] = augs_filename
    single_sample_array_dict["augmentations"] = augs_array
    if "train" in filename:
        filename_dict["train"].append(single_sample_filename_dict)
        img_dict["train"].append(single_sample_array_dict)
    if "test" in filename:
        filename_dict["test"].append(single_sample_filename_dict)
        img_dict["test"].append(single_sample_array_dict)

with open("augmentation_filenames.json", "w") as f:
    json.dump(filename_dict, f, indent=4)

with open("augmentation_arrays.pickle", "wb") as f:
    pickle.dump(img_dict, f)