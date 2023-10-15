import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

starting_point = 0.18391595229527163

res_file_list = [
    "20230606-001107_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-2-img_per_class-ADAM-SPSA-NOISY.json",
    "20230606-132732_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-3-img_per_class-ADAM-SPSA-NOISY.json",
    "20230607-160803_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA-NOISY.json",
    "20230609-190242_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-4-img_per_class-ADAM-SPSA-NOISY.json"

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
plt.axvline(x=0, linestyle='--', label = "0606-2-img", color = 'r')
plt.axvline(x=500-1, linestyle='--', label = "0606-3-img", color = 'g')
plt.axvline(x=500+500-1, linestyle='--', label = "0607-4-img", color = 'k')
plt.axvline(x=500+500+500-1, linestyle='--', label = "0609-4-img", color = 'b')
plt.axhline(y=starting_point, linestyle='--', label = "starting point", color = 'y')
plt.xlabel('iterations')
plt.ylabel('contrastive loss')
plt.title('3x3 conv, classical features, noisy')
plt.legend()
plt.savefig("losses_until_230613_file.png")