import Augmentor
import os

image_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"

# list the directories in the image folder
dirs = os.listdir(image_folder)

# for each directory, create an augmentation pipeline
for dir in dirs:
    for _ in range(10):
        p = Augmentor.Pipeline(os.path.join(image_folder, dir))
        p.rotate(probability=0.5, max_left_rotation=10, max_right_rotation=10)
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)
        p.random_distortion(probability=0.5, grid_width=8, grid_height=8, magnitude=8)
        p.skew(probability=0.5, magnitude=0.2)
        p.sample(1)