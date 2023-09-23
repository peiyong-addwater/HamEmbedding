import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

starting_point = 0.31590773737223055

res_file_list = [
    "20230606-165827_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-3-img_per_class-ADAM-SPSA-NOISY.json",
    "20230607-160832_siamese-10-class-qiskit-mnist-5x5-conv-restricted-2q-gate-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA-NOISY.json"
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
plt.axvline(x=0, linestyle='--', label = "0606-3-img", color = 'r')
plt.axvline(x=500-1, linestyle='--', label = "0606-4-img", color = 'g')
plt.axhline(y=starting_point, linestyle='--', label = "starting point", color = 'y')
plt.xlabel('iterations')
plt.ylabel('contrastive loss')
plt.title('5x5 kernel, ising-xx-yy-zz qconv, classical features, noisy')
plt.legend()
plt.savefig("losses_until_230613_file.png")