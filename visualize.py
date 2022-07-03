import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

titles = np.loadtxt("power_levels_pytorch_0.txt", dtype=str, max_rows=1)
titles = titles.tolist().split(',')

arr = []
for i in range(10):
    arr.append(np.genfromtxt(f"power_levels_keras_{i}.txt", skip_header=1, delimiter=','))
    # arr2 = np.genfromtxt("power_levels_1.txt", skip_header=1, delimiter=',')


# print(arr)
arr1 = arr[0]
arr2 = arr[4]
arr3 = arr[9]

# arr = np.genfromtxt("watt4.txt", delimiter=',', filling_values=0)

# print(arr)
# # print(np.where(arr==0)[0][0])
# # accuracy = arr[np.where(arr==0)[0][0]][0]
# # print(accuracy)
# mid = np.where(arr==0)[0][7]
# print(mid)
# arr1 = arr[:mid, :]
# arr2 = arr[mid:, :]
# print(arr1)
# print(arr2)

# arr1 = np.delete(arr1, np.where(arr1==0)[0], 0)
# arr2 = np.delete(arr2, np.where(arr2==0)[0], 0)

# arr = np.delete(arr, np.where(arr==0)[0], 0)


# print(arr[:, 0])

# print(arr)


# plt.figure(1)
# plt.plot(arr)
# # plt.figure(2)
# fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
# axs[0, 0].plot(arr[:, 0])
# axs[0, 0].set_title(titles[0])
# axs[0, 1].plot(arr[:, 1])
# axs[0, 1].set_title(titles[1])
# axs[1, 0].plot(arr[:, 2])
# axs[1, 0].set_title(titles[2])
# axs[1, 1].plot(arr[:, 3])
# axs[1, 1].set_title(titles[3])


# plt.figure(1)
# plt.plot(np.concatenate((arr1, arr2)))
# plt.plot(arr1, label="Run 1")
# plt.plot(arr2, label="Run 2")
# plt.legend()


back = min(len(arr3), len(arr2), len(arr1))-160
# plt.figure(2)
fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
axs[0, 0].plot(arr1[:, 0], label="Run 1")
axs[0, 0].plot(arr2[:, 0], label="Run 2")
axs[0, 0].plot(arr3[:, 0], label="Run 3")
axs[0, 0].axvline(x=160)
axs[0, 0].axvline(x=back)
axs[0, 0].set_title(titles[0])
axs[0, 0].legend()

axs[0, 1].plot(arr1[:, 1], label="Run 1")
axs[0, 1].plot(arr2[:, 1], label="Run 2")
axs[0, 1].plot(arr3[:, 1], label="Run 3")
axs[0, 1].axvline(x=160)
axs[0, 1].axvline(x=back)
axs[0, 1].set_title(titles[1])
axs[0, 1].legend()

axs[1, 0].plot(arr1[:, 2], label="Run 1")
axs[1, 0].plot(arr2[:, 2], label="Run 2")
axs[1, 0].plot(arr3[:, 2], label="Run 3")
axs[1, 0].axvline(x=160)
axs[1, 0].axvline(x=back)
axs[1, 0].set_title(titles[2])
axs[1, 0].legend()

axs[1, 1].plot(arr1[:, 3], label="Run 1")
axs[1, 1].plot(arr2[:, 3], label="Run 2")
axs[1, 1].plot(arr3[:, 3], label="Run 3")
axs[1, 1].axvline(x=160)
axs[1, 1].axvline(x=back)
axs[1, 1].set_title(titles[3])
axs[1, 1].legend()

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
