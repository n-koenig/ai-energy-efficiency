import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

def read_watts(path, file_name, reps):
    watts = []
    for i in range(reps):
        watts.append(np.genfromtxt(f"{path}power_levels_{file_name}_{i}.txt", skip_header=1, delimiter=','))    
    return watts
    

def read_accs(path, file_name, reps):
    accs = []
    for i in range(reps):
        accs.append(np.genfromtxt(path + file_name, skip_header=1, delimiter=',')[i*3+2])
    accs = np.asarray(accs)[:, 3]
    
    return accs


def energy_from_watts(watts):
    print(len(watts))
    energy_sum = np.asarray([np.sum(x[161:-160], axis=0) for x in watts])
    energy_integral = np.asarray([integrate.simps(x[161:-160], axis=0) for x in watts])

    return energy_integral


def linear_efficiency(avg_energy, avg_acc):
    return 1000000 * avg_acc / (avg_energy[0] + avg_energy[1] + avg_energy[2])


def barplot_energy(energy_arr, reps):
    plt.figure()
    plt.bar([x*4+0 for x in range(reps)], energy_arr[:, 0])
    plt.bar([x*4+1 for x in range(reps)], energy_arr[:, 1])
    plt.bar([x*4+2 for x in range(reps)], energy_arr[:, 2])
    plt.bar([x*4+3 for x in range(reps)], energy_arr[:, 3])


def plot_energy_and_acc(labels, avg_energy_1, avg_energy_2, avg_error_1, avg_error_2, avg_acc_1,avg_acc_2):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle("Simple Convolutional NN trained on MNIST dataset, 20 runs average")
    ax1.set_ylabel("Average energy consumptin in Joules")
    ax2.set_ylabel("Average test accuracy in %")
    ax2.set_ylim(0, 1)
    ax1.bar(labels, [avg_energy_1[0], avg_energy_2[0]], yerr=[avg_error_1[0], avg_error_2[0]], label="GPU")
    ax1.bar(labels, [avg_energy_1[1], avg_energy_2[1]], yerr=[avg_error_1[1], avg_error_2[1]], bottom=[avg_energy_1[0], avg_energy_2[0]], label="RAM")
    ax1.bar(labels, [avg_energy_1[2], avg_energy_2[2]], yerr=[avg_error_1[2], avg_error_2[2]], bottom=[avg_energy_1[0]+avg_energy_1[1], avg_energy_2[0]+avg_energy_2[1]], label="CPU")
    ax2.plot([avg_acc_1/100, avg_acc_2/100], 'ro')
    ax1.legend()


def plot_watts(titles, watts, reps):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    for j in range(4):
        index = [int(x) for x in f"{j:02b}"]
        for i in range(reps):
            axs[index[0], index[1]].plot(watts[i][:, j], label=f"Run {i}")
            axs[index[0], index[1]].set_title(titles[j])
            axs[index[0], index[1]].legend()


def tolerant_mean(arrs):
    # https://stackoverflow.com/a/59281468
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def plot_avg_watts(watts, title):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    fig.suptitle(title)
    for i in range(4):
        arr = [x[:, i] for x in watts]
        y, error = tolerant_mean(arr)
        x = np.linspace(0, y.shape[0]-1, y.shape[0])
        index = [int(x) for x in f"{i:02b}"]
        axs[index[0], index[1]].plot(y)
        axs[index[0], index[1]].fill_between(x, y-error, y+error, color='green', alpha=0.2)
        axs[index[0], index[1]].set_title(titles[i])
        

reps = 5
paths = ['dump/', 'MNIST_CNN/1/']
path = paths[0]
titles = ['nvml:nvidia_geforce_gtx_970_0', 'rapl:ram', 'rapl:cores', 'rapl:pkg']

# get energy
keras_watts = read_watts(path, 'keras', reps)
#pytorch_watts = read_watts(path, 'pytorch', reps)
keras_energy = energy_from_watts(keras_watts)
# pytorch_energy = energy_from_watts(pytorch_watts)

# compute avg and std for energy
keras_energy_avg = np.average(keras_energy, axis=0)
keras_energy_std = 100 * np.std(keras_energy, axis=0)/keras_energy_avg
# pytorch_energy_avg = np.average(pytorch_energy, axis=0)
# pytorch_energy_std = 100 * np.std(pytorch_energy, axis=0)/pytorch_energy_avg

# get accuracy
# keras_accs = np.average(read_accs(path, 'log_keras.csv', reps))
p# ytorch_accs = np.average(read_accs(path, 'log_pytorch.csv', reps))

# get avg and std for energy
# keras_acc_avg = 100*np.average(keras_accs)
# keras_acc_std = 100*np.std(keras_accs)
# pytorch_acc_avg = np.average(pytorch_accs)
# pytorch_acc_std = np.std(pytorch_accs)

# calculate linear efficiency
# keras_efficiency = linear_efficiency(keras_energy_avg, keras_acc_avg)
# pytorch_efficiency = linear_efficiency(pytorch_energy_avg, pytorch_acc_avg)

# print calculation results 
# print(', '.join(f'avg: {x:,.2f}' for x in keras_energy_avg))
# print(', '.join(f'std: {x:.2f}%' for x in keras_energy_std))
# print(', '.join(f'avg: {x:,.2f}' for x in pytorch_energy_avg))
# print(', '.join(f'std: {x:.2f}%' for x in pytorch_energy_std))
# print(f"{keras_acc_avg:.2f}%, {pytorch_acc_avg:.2f}%")
# print(f"{keras_acc_std:.2f}%, {pytorch_acc_std:.2f}%")

# barplot_energy(keras_energy, reps)
# labels = [f'keras\n(efficiency: {keras_efficiency:.2f})', f'pytorch\n(efficiency: {pytorch_efficiency:.2f})']
# print(keras_acc_avg, pytorch_acc_avg)
# plot_energy_and_acc(labels, keras_energy_avg, pytorch_energy_avg, keras_energy_std, pytorch_energy_std, keras_acc_avg, pytorch_acc_avg)
plot_watts(titles, keras_watts, reps)
# plot_watts(titles, pytorch_watts, reps)
# plot_avg_watts(keras_watts, "Simple Keras Convolutional NN trained on MNIST dataset, 20 runs average")
# plot_avg_watts(pytorch_watts, "Simple Pytorch Convolutional NN trained on MNIST dataset, 20 runs average")


plt.show()