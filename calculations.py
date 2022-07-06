import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

class ExperimentData:
    def __init__(self, name, reps, epochs):
        self.name = name
        self.reps = reps
        self.epochs = epochs
        self.watts = []
        self.energy = []
        self.energy_avg = []
        self.energy_std = []
        self.acc_avg = 0
        self.acc_std = 0
        self.efficiency = 0

    def set_energy_data(self, path, filename):
        self.watts = read_watts(path, filename, self.reps)
        self.energy = energy_from_watts(self.watts)
        self.energy_avg = np.average(self.energy, axis=0)
        self.energy_std = 100 * np.std(self.energy, axis=0)/self.energy_avg

    def set_acc_data(self, path, filename):
        self.accs = read_accs(path, filename, self.reps, self.epochs)
        self.acc_avg = 100*np.average(self.accs)
        self.acc_std = 100*np.std(self.accs)

    def set_efficiency_data(self):
        self.efficiency = linear_efficiency(self.energy_avg, self.acc_avg)

    def print_data(self):
        print(f"{self.name} Experiment Data:")
        print(f"Average energy consumption <Joules (deviation %)>:")
        print(f"GPU: {self.energy_avg[0]:,.2f} ({self.energy_std[0]:.2f}%)")
        print(f"RAM: {self.energy_avg[1]:,.2f} ({self.energy_std[1]:.2f}%)")
        print(f"CPU: {self.energy_avg[2]:,.2f} ({self.energy_std[2]:.2f}%)")
        print(f"PKG: {self.energy_avg[3]:,.2f} ({self.energy_std[3]:.2f}%)")
        print(f"Average accuracy <accuracy (deviation %)>:")
        print(f"{self.acc_avg:.2f} ({self.acc_std:.2f}%)")
        print(f"Efficiency: {self.efficiency}")


def read_watts(path, file_name, reps):
    watts = []
    for i in range(reps):
        watts.append(np.genfromtxt(f"{path}power_levels_{file_name}_{i}.txt", skip_header=1, delimiter=','))    
    return watts
    

def read_accs(path, file_name, reps, epochs):
    accs = []
    for i in range(reps):
        accs.append(np.genfromtxt(f"{path}log_{file_name}.csv", skip_header=1, delimiter=',')[(i+1)*epochs-1])
    accs = np.asarray(accs)[:, 3]
    
    return accs


def energy_from_watts(watts):
    energy_sum = np.asarray([np.sum(x[161:-160], axis=0) for x in watts])
    energy_integral = np.asarray([integrate.simps(x[161:-160], axis=0) for x in watts])

    return energy_integral


def linear_efficiency(avg_energy, avg_acc):
    return 1000000 * avg_acc / (avg_energy[0] + avg_energy[1] + avg_energy[2])


def barplot_energy(title, energy_arr, reps):
    plt.figure()
    plt.title(title)
    plt.bar([x*4+0 for x in range(reps)], energy_arr[:, 0])
    plt.bar([x*4+1 for x in range(reps)], energy_arr[:, 1])
    plt.bar([x*4+2 for x in range(reps)], energy_arr[:, 2])
    plt.bar([x*4+3 for x in range(reps)], energy_arr[:, 3])


def plot_energy_and_acc(title, labels, avg_energy_1, avg_error_1, avg_acc_1):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle(title)
    ax1.set_ylabel("Average energy consumptin in Joules")
    ax2.set_ylabel("Average test accuracy in %")
    ax2.set_ylim(0, 1)
    ax1.bar(labels, [avg_energy_1[0]], yerr=[avg_error_1[0]], label="GPU")
    ax1.bar(labels, [avg_energy_1[1]], yerr=[avg_error_1[1]], bottom=[avg_energy_1[0]], label="RAM")
    ax1.bar(labels, [avg_energy_1[2]], yerr=[avg_error_1[2]], bottom=[avg_energy_1[0]+avg_energy_1[1]], label="CPU")
    ax2.plot([avg_acc_1/100], 'ro')
    ax1.legend()


def plot_compare_energy_and_acc(title, labels, avg_energy_1, avg_energy_2, avg_error_1, avg_error_2, avg_acc_1,avg_acc_2):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle(title)
    ax1.set_ylabel("Average energy consumptin in Joules")
    ax2.set_ylabel("Average test accuracy in %")
    ax2.set_ylim(0, 1)
    ax1.bar(labels, [avg_energy_1[0], avg_energy_2[0]], yerr=[avg_error_1[0], avg_error_2[0]], label="GPU")
    ax1.bar(labels, [avg_energy_1[1], avg_energy_2[1]], yerr=[avg_error_1[1], avg_error_2[1]], bottom=[avg_energy_1[0], avg_energy_2[0]], label="RAM")
    ax1.bar(labels, [avg_energy_1[2], avg_energy_2[2]], yerr=[avg_error_1[2], avg_error_2[2]], bottom=[avg_energy_1[0]+avg_energy_1[1], avg_energy_2[0]+avg_energy_2[1]], label="CPU")
    ax2.plot([avg_acc_1/100, avg_acc_2/100], 'ro')
    ax1.legend()


def plot_watts(title, titles, watts, reps):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    fig.suptitle(title)
    l = list()
    for j in range(4):
        index = [int(x) for x in f"{j:02b}"]
        for i in range(reps):
            l.append(axs[index[0], index[1]].plot(watts[i][:, j]))
            axs[index[0], index[1]].set_title(titles[j])
    fig.legend(l, labels=[f"Run {i}" for i in range(20)], loc="right")


def tolerant_mean(arrs):
    # https://stackoverflow.com/a/59281468
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def plot_avg_watts(title, watts):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    fig.suptitle(title)
    l = list()
    for i in range(4):
        arr = [x[:, i] for x in watts]
        y, error = tolerant_mean(arr)
        x = np.linspace(0, y.shape[0]-1, y.shape[0])
        index = [int(x) for x in f"{i:02b}"]
        l.append(axs[index[0], index[1]].plot(y))
        axs[index[0], index[1]].fill_between(x, y-error, y+error, color='green', alpha=0.2)
        axs[index[0], index[1]].set_title(titles[i])
    fig.legend(l, labels=['average', 'standard\ndeviation'], loc="right")
        

paths = ['dump/', 'MNIST_CNN/1/', 'MNIST_CNN/3/']
titles = ['nvml:nvidia_geforce_gtx_970_0', 'rapl:ram', 'rapl:cores', 'rapl:pkg']

keras_data = ExperimentData('keras', 20, 12)
keras_data.set_energy_data(paths[2], 'keras')
keras_data.set_acc_data(paths[2], 'keras')
keras_data.set_efficiency_data()

pytorch_data = ExperimentData('pytorch', 20, 12)
pytorch_data.set_energy_data(paths[2], 'pytorch')
pytorch_data.set_acc_data(paths[2], 'pytorch')
pytorch_data.set_efficiency_data()

keras_data.print_data()
print("-----------------------------------------------------")
pytorch_data.print_data()

# barplot_energy("title", keras_energy, reps)
labels = [f'keras\n(efficiency: {keras_data.efficiency:.2f})', f'pytorch\n(efficiency: {pytorch_data.efficiency:.2f})']
plot_compare_energy_and_acc(f"Simple Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", labels, keras_data.energy_avg, pytorch_data.energy_avg, keras_data.energy_std, pytorch_data.energy_std, keras_data.acc_avg, pytorch_data.acc_avg)
plot_energy_and_acc(f"Simple Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", labels[0], keras_data.energy_avg, keras_data.energy_std, keras_data.acc_avg)
plot_watts(f"Simple {keras_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs", titles, keras_data.watts, keras_data.reps)
plot_watts(f"Simple {pytorch_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs", titles, pytorch_data.watts, pytorch_data.reps)
plot_avg_watts(f"Simple {keras_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", keras_data.watts)
plot_avg_watts(f"Simple {pytorch_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", pytorch_data.watts)


plt.show()