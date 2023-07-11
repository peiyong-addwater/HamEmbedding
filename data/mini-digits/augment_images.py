import random

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

import albumentations as A
import os

image_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
example_save_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/example_augmentation"
augmentation_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/augmented_images"
labels = [0,1,2,3,4,5,6,7,8,9]

# list image filenames
img_files = os.listdir(image_folder)
random.shuffle(img_files)
print(img_files[:10])

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


# augment all images
NUM_AUGMENTATIONS = 10
for filename in tqdm(img_files):
    img = cv2.imread(os.path.join(image_folder, filename))
    cv2.imwrite(os.path.join(image_folder, filename), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    for i in range(NUM_AUGMENTATIONS):
        transformed = transform(image=img)
        transformed_image = transformed["image"]

        save_name = f"{filename.split('.')[0]}_aug_{i}.png"
        cv2.imwrite(os.path.join(augmentation_folder, save_name), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2GRAY))