import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

reps = 3

keras_watts = []
keras_accs = []
pytorch_watts = []
pytorch_accs = []
for i in range(3):
    keras_watts.append(np.genfromtxt(f"power_levels_keras_{i}.txt", skip_header=1, delimiter=','))
    pytorch_watts.append(np.genfromtxt(f"power_levels_pytorch_{i}.txt", skip_header=1, delimiter=','))
    keras_accs.append(np.genfromtxt(f"log_keras.csv", skip_header=1, delimiter=',')[i*3+2])
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

plt.figure(2)
plt.bar([x*4+0 for x in range(reps)], pytorch_energy_1[:, 0])
plt.bar([x*4+1 for x in range(reps)], pytorch_energy_1[:, 1])
plt.bar([x*4+2 for x in range(reps)], pytorch_energy_1[:, 2])
plt.bar([x*4+3 for x in range(reps)], pytorch_energy_1[:, 3])

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