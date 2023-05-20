import json
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 22})

res_file_list = ["20230510-134752_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-2-img_per_class-COBYLA.json",
                 "20230512-030158_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-2-img_per_class-ADAM-SPSA.json",
                 "20230515-010140_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-2-img_per_class-ADAM-SPSA.json",
                 "20230516-153220_siamese-10-class-qiskit-mnist-3x3-conv-classical-features-tiny-image-results-3-img_per_class-ADAM-SPSA.json"]

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
plt.axvline(x=500-1,  linestyle='--', label = "0510",color = 'b')
plt.axvline(x=1000+500-1, linestyle='--', label = "0512", color = 'r')
plt.axvline(x=1000+1000+500-1, linestyle='--', label = "0515", color = 'g')
plt.axvline(x=1000+1000+1000+500-1, linestyle='--', label = "0516", color = 'k')
plt.xlabel('iterations')
plt.ylabel('contrastive loss (classical measurement as features)')
plt.legend()
plt.savefig("losses_until_230520_file.png")