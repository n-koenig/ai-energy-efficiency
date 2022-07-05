import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

reps = 20

keras_watts = []
keras_accs = []
pytorch_watts = []
pytorch_accs = []
for i in range(reps):
    keras_watts.append(np.genfromtxt(f"power_levels_keras_{i}.txt", skip_header=1, delimiter=','))
    keras_accs.append(np.genfromtxt(f"log_keras.csv", skip_header=1, delimiter=',')[i*12+11])
    pytorch_watts.append(np.genfromtxt(f"power_levels_pytorch_{i}.txt", skip_header=1, delimiter=','))
    pytorch_accs.append(np.genfromtxt(f"log_pytorch.csv", skip_header=1, delimiter=',')[i*3+2])


keras_accs = np.asarray(keras_accs)[:, 3]
pytorch_accs = np.asarray(pytorch_accs)[:, 3]



keras_energy_1 = np.asarray([np.sum(x[161:-160], axis=0) for x in keras_watts])
keras_energy_2 = np.asarray([np.sum(x, axis=0) - np.sum(x[:161], axis=0) - np.sum(x[-160:], axis=0) for x in keras_watts])
keras_energy_3 = np.asarray([integrate.simps(x[161:-160], axis=0) for x in keras_watts])
# print(keras_energy_1)
# print(keras_energy_2[0])
# print(keras_energy_3[0])

plt.figure(1)
plt.bar([x*4+0 for x in range(reps)], keras_energy_1[:, 0])
plt.bar([x*4+1 for x in range(reps)], keras_energy_1[:, 1])
plt.bar([x*4+2 for x in range(reps)], keras_energy_1[:, 2])
plt.bar([x*4+3 for x in range(reps)], keras_energy_1[:, 3])
plt.tick_params()
pytorch_energy_1 = np.asarray([np.sum(x[161:-160], axis=0) for x in pytorch_watts], dtype=float)
pytorch_energy_2 = np.asarray([np.sum(x, axis=0) - np.sum(x[:161], axis=0) - np.sum(x[-160:], axis=0) for x in pytorch_watts])
pytorch_energy_3 = np.asarray([integrate.simps(x[161:-160], axis=0) for x in pytorch_watts])
# print(pytorch_energy_1[0])
# print(pytorch_energy_2[0])
# print(pytorch_energy_3[0])

# plt.figure(2)
# plt.bar([x*4+0 for x in range(reps)], pytorch_energy_1[:, 0])
# plt.bar([x*4+1 for x in range(reps)], pytorch_energy_1[:, 1])
# plt.bar([x*4+2 for x in range(reps)], pytorch_energy_1[:, 2])
# plt.bar([x*4+3 for x in range(reps)], pytorch_energy_1[:, 3])



# plt.show()

keras_avg = np.average(keras_energy_1, axis=0)
keras_std = 100 * np.std(keras_energy_1, axis=0)/keras_avg
pytorch_avg = np.average(pytorch_energy_1, axis=0)
pytorch_std = 100 * np.std(pytorch_energy_1, axis=0)/pytorch_avg

print(', '.join(f'avg: {q:,.2f}' for q in keras_avg))
print(', '.join(f'std: {q:.2f}%' for q in keras_std))

print(', '.join(f'avg: {q:,.2f}' for q in pytorch_avg))
print(', '.join(f'std: {q:.2f}%' for q in pytorch_std))

keras_acc_avg = 100*np.average(keras_accs)
keras_acc_std = 100*np.std(keras_accs)
pytorch_acc_avg = np.average(pytorch_accs)
pytorch_acc_std = np.std(pytorch_accs)

print(f"{keras_acc_avg:.2f}%, {pytorch_acc_avg:.2f}%")
print(f"{keras_acc_std:.2f}%, {pytorch_acc_std:.2f}%")

keras_eff = keras_acc_avg / (keras_avg[0] + keras_avg[1] + keras_avg[2])
pytorch_eff = pytorch_acc_avg / (pytorch_avg[0] + pytorch_avg[1] + pytorch_avg[2])

labels = [f'keras\n(efficiency: {1000000*keras_eff:.2f})', f'pytorch\n(efficiency: {1000000*pytorch_eff:.2f})']
fig, ax1 = plt.subplots()
ax1.bar(labels, [keras_avg[0], pytorch_avg[0]], yerr=[keras_std[0], pytorch_std[0]], label="GPU")
ax1.bar(labels, [keras_avg[1], pytorch_avg[1]], yerr=[keras_std[1], pytorch_std[1]], bottom=[keras_avg[0], pytorch_avg[0]], label="RAM")
ax1.bar(labels, [keras_avg[2], pytorch_avg[2]], yerr=[keras_std[2], pytorch_std[2]], bottom=[keras_avg[0]+keras_avg[1], pytorch_avg[0]+pytorch_avg[1]], label="CPU")
# ax1.set_ylim(0, 350000000)
ax1.set_ylabel("Average energy consumptin in Joules")
ax1.legend()
ax2 = ax1.twinx()
ax2.set_ylim(0, 1)
ax2.set_ylabel("Average test accuracy in %")
ax2.plot([keras_acc_avg/100, pytorch_acc_avg/100], 'ro')
fig.suptitle("Simple Convolutional NN trained on MNIST dataset, 20 runs average")
plt.show()