from matplotlib import rcParams
import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import math

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

    def set_energy_data(self, path, offset):
        self.watts = read_watts(path, self.name, self.reps)
        self.energy = energy_from_watts(self.watts, offset)
        self.energy_avg = np.average(self.energy, axis=0)
        self.energy_std = 100 * np.std(self.energy, axis=0)/self.energy_avg

    def set_acc_data(self, path):
        self.accs = read_accs(path, self.name, self.reps, self.epochs)
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


def energy_from_watts(watts, offset):
    energy_sum = np.asarray([np.sum(x[offset+1:-offset], axis=0) for x in watts])
    energy_integral = np.asarray([integrate.simps(x[offset+1:-offset], axis=0) for x in watts])

    return energy_integral


def linear_efficiency(avg_energy, avg_acc):
    return (avg_energy[0] + avg_energy[1] + avg_energy[2]) / avg_acc
    return 1000000 * avg_acc / (avg_energy[0] + avg_energy[1] + avg_energy[2])


def barplot_energy(title, energy_arr, reps):
    plt.figure()
    plt.title(title)
    plt.bar([x*4+0 for x in range(reps)], energy_arr[:, 0])
    plt.bar([x*4+1 for x in range(reps)], energy_arr[:, 1])
    plt.bar([x*4+2 for x in range(reps)], energy_arr[:, 2])
    plt.bar([x*4+3 for x in range(reps)], energy_arr[:, 3])


def plot_compare_energy_and_acc(title, labels, avg_energy_list, avg_error_list, avg_acc_list, show_line=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    fig.suptitle(title)
    l = []
    l.append(ax1.set_ylabel("Energy consumptin [kJ]"))
    l.append(ax2.set_ylabel("Test accuracy [%]"))
    ax2.set_ylim(0, 1)
    ax1.bar(labels, [x[0] for x in avg_energy_list], yerr=[x[0] for x in avg_error_list], label="GPU")
    ax1.bar(labels, [x[1] for x in avg_energy_list], yerr=[x[1] for x in avg_error_list], bottom=[x[0] for x in avg_energy_list], label="RAM")
    ax1.bar(labels, [x[2] for x in avg_energy_list], yerr=[x[2] for x in avg_error_list], bottom=[(x[0]+x[1]) for x in avg_energy_list], label="CPU")
    # ax1.plot([(x[0]+x[1]+x[2]) for x in avg_energy_list], 'yo', label="Total energy")
    # ax1.plot([(x[0]+x[1]+x[2]) for x in avg_energy_list], 'y')
    ax2.plot([x/100 for x in avg_acc_list], 'ro')
    if show_line:
        ax2.plot([x/100 for x in avg_acc_list], 'r')
    ax1.legend()
    ax2.legend(['Accuracy'])
    # fig.legend(['GPU', 'RAM', 'CPU', 'Accuracy'])


def plot_watts(title, titles, watts, reps, offset):
    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
    fig.suptitle(title)
    l = []
    for j in range(4):
        index = [int(x) for x in f"{j:02b}"]
        for i in range(reps):
            l.append(axs[index[0], index[1]].plot(watts[i][:, j]))
            axs[index[0], index[1]].set_title(titles[j])
        axs[index[0], index[1]].axvline(offset)
        axs[index[0], index[1]].axvline(max([len(x) for x in watts])-offset)
    for a in fig.axes:
        a.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        a.tick_params(axis='y', which='both', bottom=True, top=False, labelleft=True)
    fig.text(0.5, 0.04, 'Sample number', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power level in mW', ha='center', va='center', rotation='vertical')
    fig.legend(l, labels=[f"Run {i}" for i in range(20)], loc="right")

def tolerant_mean(arrs):
    # https://stackoverflow.com/a/59281468
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def plot_avg_watts(title, watts, offset):
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
        axs[index[0], index[1]].axvline(offset)
        axs[index[0], index[1]].axvline(len(y)-offset)
    for a in fig.axes:
        a.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        a.tick_params(axis='y', which='both', bottom=True, top=False, labelleft=True)
    fig.text(0.5, 0.04, 'Sample number', ha='center', va='center')
    fig.text(0.06, 0.5, 'Power level in mW', ha='center', va='center', rotation='vertical')
    fig.legend(l, labels=['average', 'standard\ndeviation'], loc="right")
        

def eval_data_load_exp():
    keras_data_loads = []
    for i in range(10):
        keras_data_loads.append(ExperimentData(f'keras_{(i+1)*10}', 20, 12))
        keras_data_loads[i].set_energy_data(paths[3], 400)
        keras_data_loads[i].set_acc_data(paths[3])
        keras_data_loads[i].set_efficiency_data()
    for i in range(10):
        keras_data_loads[i].print_data()
        print("-----------------------------------------------------")

    labels = [f'keras {(i+1)*10}\nefficiency: {keras_data_loads[i].efficiency:.2f}' for i in range(10)]
    # plot_compare_energy_and_acc(f"Simple Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", labels, 
    #                            [keras_data_loads[i].energy_avg for i in range(10)], 
    #                            [keras_data_loads[i].energy_std for i in range(10)], 
    #                            [keras_data_loads[i].acc_avg for i in range(10)],
    #                            True)
    # plt.savefig("./efficiency_comparison.png")

    for i in range(10):
        continue
        plot_watts(f"Simple keras Convolutional NN trained on MNIST dataset with {(i+1)*10} % training data, {keras_data.reps} runs", titles, keras_data_loads[i].watts, keras_data_loads[i].reps, 400)
        plt.savefig(f"./keras_{(i+1)*10}_watts.png", dpi=199)
        plot_avg_watts(f"Simple keras Convolutional NN trained on MNIST dataset with {(i+1)*10} % training data, {keras_data.reps} runs average", keras_data_loads[i].watts, 400)
        plt.savefig(f"./keras_{(i+1)*10}_watts_average.png", dpi=199)

    # efficiency_test([x.energy_avg for x in keras_data_loads], [x.acc_avg for x in keras_data_loads])
    return keras_data_loads


def efficiency_test(keras_data_loads, pytorch_data):
    energy_sum = [np.sum(x.energy_avg[0:3])/1000 for x in keras_data_loads]
    energy_sum.append(np.sum(pytorch_data.energy_avg[0:3])/1000)
    acc = [x.acc_avg for x in keras_data_loads]
    acc.append(pytorch_data.acc_avg)

    print(energy_sum, acc)
    x = np.linspace(0, energy_sum[-1], 11)
    y = -208 + 24 * np.log(x)
    print(x, y)

    plt.scatter(energy_sum, acc)
    plt.plot(x, y)

    energy = []
    acc_2 = []
    for i in range(1, 11):
        energy.append((energy_sum[i] - energy_sum[i-1])/(energy_sum[-1]-energy_sum[0]))
        acc_2.append(acc[i] - acc[i-1])

    print(energy, acc_2)

    fig, ax = plt.subplots()
    ax.scatter(energy, acc_2)
    [ax.text(energy[i], acc_2[i], str(i)) for i in range(10)]
    # ax.set_xlim((0, 120000))
    # ax.plot(energy, acc_2)


    fig, ax = plt.subplots()
    eff = [((acc[i])/energy_sum[i]) for i in range(11)]
    eff_2 = [((acc[i]**2)/energy_sum[i]) for i in range(11)]
    eff_3 = [((acc[i]**4)/energy_sum[i]) for i in range(11)]
    eff_4 = [((math.exp(acc[i]/100))/energy_sum[i]) for i in range(11)]

    print(eff_4)

    eff = (eff - np.min(eff)) / (np.max(eff) - np.min(eff))
    eff_2 = (eff_2 - np.min(eff_2)) / (np.max(eff_2) - np.min(eff_2))
    eff_3 = (eff_3 - np.min(eff_3)) / (np.max(eff_3) - np.min(eff_3))
    eff_4 = (eff_4 - np.min(eff_4)) / (np.max(eff_4) - np.min(eff_4))

    # plt.scatter(np.linspace(0, 10, 11), eff)
    ax.plot(eff)
    ax.plot(eff_2)
    ax.plot(eff_3)
    # ax.plot(eff_4)
    plt.show()



paths = ['dump/', 'MNIST_CNN/1/', 'MNIST_CNN/2/', 'MNIST_CNN/3/']
titles = ['nvml:nvidia_geforce_gtx_970_0', 'rapl:ram', 'rapl:cores', 'rapl:pkg']

keras_data = ExperimentData('keras', 20, 12)
keras_data.set_energy_data(paths[1], 180)
keras_data.set_acc_data(paths[1])
keras_data.set_efficiency_data()

pytorch_data = ExperimentData('pytorch', 20, 12)
pytorch_data.set_energy_data(paths[1], 180)
pytorch_data.set_acc_data(paths[1])
pytorch_data.set_efficiency_data()

keras_data_loads = eval_data_load_exp()

keras_data.print_data()
print("-----------------------------------------------------")
pytorch_data.print_data()

# efficiency_test(keras_data_loads, pytorch_data)

# barplot_energy("title", keras_energy, reps)

labels = [f'keras\n(efficiency: {keras_data.efficiency:.2f})', f'pytorch\n(efficiency: {pytorch_data.efficiency:.2f})']
labels = ['keras', 'pytorch']
plot_compare_energy_and_acc(f"Simple Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average",
                            labels,
                            [keras_data.energy_avg, pytorch_data.energy_avg],
                            [keras_data.energy_std, pytorch_data.energy_std],
                            [keras_data.acc_avg, pytorch_data.acc_avg],
                            True)

plot_watts(f"Simple {keras_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs", titles, keras_data.watts, keras_data.reps, 160)
plot_watts(f"Simple {pytorch_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs", titles, pytorch_data.watts, pytorch_data.reps, 160)
plot_avg_watts(f"Simple {keras_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", keras_data.watts, 160)
plot_avg_watts(f"Simple {pytorch_data.name} Convolutional NN trained on MNIST dataset, {keras_data.reps} runs average", pytorch_data.watts, 160)
plt.show()
