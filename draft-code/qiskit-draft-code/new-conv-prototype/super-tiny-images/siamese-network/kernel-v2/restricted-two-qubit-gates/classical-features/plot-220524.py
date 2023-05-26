import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

res_file_list = [
    "20230521-134040_siamese-10-class-qiskit-mnist-5x5-conv-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA.json",
    "20230522-090623_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA.json",
    "20230523-110519_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA.json",
    "20230524-131309_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-8-img_per_class-ADAM-SPSA.json"
]
losses = []
for res_file in res_file_list:
    with open(res_file) as json_file:
        data = json.load(json_file)
        print(res_file)
        print("Iterations: ", len(data['losses']))
        losses.extend(data['losses'])

print(len(losses))
iter = list(range(len(losses)))


fig = plt.figure(figsize=(16, 9))
plt.plot(iter, losses)
plt.axvline(x=1000-1, linestyle='--', label = "0521", color = 'r')
plt.axvline(x=1000+1000-1, linestyle='--', label = "0522", color = 'g')
plt.axvline(x=1000+1000+1000-1, linestyle='--', label = "0523", color = 'k')
plt.axvline(x=2000+1000+1000+1000-1, linestyle='--', label = "0524", color = 'b')
plt.xlabel('iterations')
plt.ylabel('contrastive loss (classical measurement as features)')
plt.legend()
plt.savefig("losses_until_230525_file.png")