import numpy as np
from urllib import request
import gzip
import pickle
from PIL import Image
import pandas as pd
import os
import csv

img_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits/images"
save_dir = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/mini-digits"

image_filename = os.listdir(img_dir)
print(image_filename)