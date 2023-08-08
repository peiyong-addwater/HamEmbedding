import numpy as np
from urllib import request
import gzip
import pickle
from PIL import Image
import pandas as pd
import os
import csv
import re

img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
save_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits"
csv_filename = "annotated_labels.csv"

image_filename = os.listdir(img_dir)
annotations = []
for filename in image_filename:
    m = re.search('_label_(.+?).png', filename)
    if m:
        label=int(m.group(1))
        annotations.append([filename, label])

with open(csv_filename, 'w') as f:
    writer = csv.writer(f)
    writer.writerows(annotations)

print("Save complete.")
print(os.listdir(save_dir))
print(csv_filename in os.listdir(save_dir))
print(csv_filename)