import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

reps = 10

titles = np.loadtxt("first_experiments/power_levels_pytorch_0.txt", dtype=str, max_rows=1)
titles = titles.tolist().split(',')

arr = []
for i in range(reps):
    arr.append(np.genfromtxt(f"first_experiments/power_levels_pytorch_{i}.txt", skip_header=1, delimiter=','))
    # arr2 = np.genfromtxt("power_levels_1.txt", skip_header=1, delimiter=',')

print(len(arr))

# plt.figure(1)
# plt.plot(np.concatenate((arr1, arr2)))
# plt.plot(arr1, label="Run 1")
# plt.plot(arr2, label="Run 2")
# plt.legend()

fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
for i in range(1, reps):
    print(i)
    axs[0, 0].plot(arr[i][:, 0], label=f"Run {i}")
    axs[0, 0].set_title(titles[0])
    axs[0, 0].legend()

    axs[0, 1].plot(arr[i][:, 1], label=f"Run {i}")
    axs[0, 1].set_title(titles[1])
    axs[0, 1].legend()

    axs[1, 0].plot(arr[i][:, 2], label=f"Run {i}")
    axs[1, 0].set_title(titles[2])
    axs[1, 0].legend()

    axs[1, 1].plot(arr[i][:, 3], label=f"Run {i}")
    axs[1, 1].set_title(titles[3])
    axs[1, 1].legend()




arr = np.asarray(arr)

arr0 = []
arr1 = []
arr2 = []
arr3 = []
for i in range(1, reps):
    arr0.append(arr[i][:, 0])
    arr1.append(arr[i][:, 1])
    arr2.append(arr[i][:, 2])
    arr3.append(arr[i][:, 3])

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
fig.suptitle("Simple Pytorch Convolutional NN trained on MNIST dataset, 20 runs average")
y, error = tolerant_mean(arr0)
x = np.linspace(0, y.shape[0]-1, y.shape[0])
axs[0, 0].plot(y)
axs[0, 0].fill_between(x, y-error, y+error, color='green', alpha=0.2)
axs[0, 0].set_title(titles[0])
y, error = tolerant_mean(arr1)
axs[0, 1].plot(y)
axs[0, 1].set_title(titles[1])
axs[0, 1].fill_between(x, y-error, y+error, color='green', alpha=0.2)
y, error = tolerant_mean(arr2)
axs[1, 0].plot(y)
axs[1, 0].fill_between(x, y-error, y+error, color='green', alpha=0.2)
axs[1, 0].set_title(titles[2])
y, error = tolerant_mean(arr3)
axs[1, 1].plot(y)
axs[1, 1].fill_between(x, y-error, y+error, color='green', alpha=0.2)
axs[1, 1].set_title(titles[3])






# back = min(len(arr3), len(arr2), len(arr1))-160
# plt.figure(2)
# axs[0, 0].plot(arr1[:, 0], label="Run 1")
# axs[0, 0].plot(arr2[:, 0], label="Run 2")
# axs[0, 0].plot(arr3[:, 0], label="Run 3")
# axs[0, 0].axvline(x=160)
# axs[0, 0].axvline(x=back)
# axs[0, 0].set_title(titles[0])
# axs[0, 0].legend()

# axs[0, 1].plot(arr1[:, 1], label="Run 1")
# axs[0, 1].plot(arr2[:, 1], label="Run 2")
# axs[0, 1].plot(arr3[:, 1], label="Run 3")
# axs[0, 1].axvline(x=160)
# axs[0, 1].axvline(x=back)
# axs[0, 1].set_title(titles[1])
# axs[0, 1].legend()

# axs[1, 0].plot(arr1[:, 2], label="Run 1")
# axs[1, 0].plot(arr2[:, 2], label="Run 2")
# axs[1, 0].plot(arr3[:, 2], label="Run 3")
# axs[1, 0].axvline(x=160)
# axs[1, 0].axvline(x=back)
# axs[1, 0].set_title(titles[2])
# axs[1, 0].legend()

# axs[1, 1].plot(arr1[:, 3], label="Run 1")
# axs[1, 1].plot(arr2[:, 3], label="Run 2")
# axs[1, 1].plot(arr3[:, 3], label="Run 3")
# axs[1, 1].axvline(x=160)
# axs[1, 1].axvline(x=back)
# axs[1, 1].set_title(titles[3])
# axs[1, 1].legend()

plt.show()



# print(arr[0][161:-160])
# watts1 = np.sum(arr[0][161:-160], axis=0)
# watts2 = np.sum(arr[1][161:-160], axis=0)
# print(watts1)
# print(watts2)
# print(np.sum(arr[0], axis=0))
# print(np.sum(arr[1], axis=0))

# # print(arr[0][:161])
# # print(arr[0][-160:])
# # print(np.concatenate((arr[0][:160], arr[0][:-161])))
# alt_watts1 = np.sum(arr[0], axis=0) - np.sum(arr[0][:161], axis=0) - np.sum(arr[0][-160:], axis=0)
# alt_watts2 = np.sum(arr[1], axis=0) - np.sum(arr[1][:161], axis=0) - np.sum(arr[1][-160:], axis=0)
# # print(alt_watts1)
# # print(alt_watts2)

# integral1 = integrate.simps(arr[0][161:-160], axis=0)
# print(integral1)
# # print(f"{integrate.simps(arr[0][161:-160], axis=0))
