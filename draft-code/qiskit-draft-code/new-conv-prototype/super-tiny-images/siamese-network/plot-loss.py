import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

res_file = "siamese-10-class-qiskit-mnist-5x5-conv-multiclass-tiny-image-results-3-img_per_class-COBYLA.json"
with open(res_file, 'r') as f:
    res_dict = json.load(f)

sns.set_style('whitegrid')
colors = sns.color_palette()

loss = res_dict['losses']
iter = list(range(len(loss)))

fig = plt.figure(figsize=(16, 9))
plt.plot(iter, loss)
plt.xlabel('iterations')
plt.ylabel('contrastive loss')
plt.savefig(res_file+".png")