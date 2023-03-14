import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

res_file = "siamese-10-class-qiskit-mnist-5x5-conv-multiclass-tiny-image-results-3-img_per_class-COBYLA.json"
with open(res_file, 'r') as f:
    res_dict = json.load(f)

