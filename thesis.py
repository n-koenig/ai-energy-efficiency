import numpy as np
import matplotlib.pyplot as plt


class ExperimentData:
    def __init__(self, name, reps):
        self.name = name
        self.reps = reps
        self.watts = []
        self.energy = []
        self.energy_avg = []
        self.energy_std = []
        self.acc_avg = 0
        self.acc_std = 0
        self.efficiency = 0


    def set_energy_data(self, path, offset, interval=50):
        self.watts = read_watts(path, self.name, self.reps)
        print(self.watts)
        self.energy = energy_from_watts(self.watts, offset, interval)
        print(self.energy)
        self.energy_avg = np.average(self.energy, axis=0)
        print(self.energy_avg)
        self.energy_std = 100 * np.std(self.energy, axis=0)/self.energy_avg


def read_watts(path, file_name, reps):
    watts = []
    for i in range(reps):
        watts.append(np.genfromtxt(f"{path}power_levels_{file_name}_{i}.txt", skip_header=1, delimiter=','))    
    return watts

def energy_from_watts(watts, offset, interval):
    # energy_sum = np.asarray([np.sum(x[offset+1:-offset]/1000, axis=0) * (interval/1000) for x in watts])
    energy_sum = np.asarray([np.sum(x[offset+1:-offset], axis=0)/(1000000/interval) for x in watts])
    return energy_sum


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
    # fig.legend(l, labels=[f"Run {i}" for i in range(20)], loc="right")


def fig_data_prep():
    data = ExperimentData('keras_10', 3)
    data.set_energy_data('MNIST_CNN/5/', 200, 100)

    plt.figure()
    plt.title('GPU power draw')
    for i in range(3):
        plt.plot(data.watts[i][:, 0]/1000, label=f'Run {i}')
    plt.axvline(x=200, color='r')
    plt.axvline(x=np.average([len(x) for x in data.watts])-200, color='r')
    x = np.linspace(0, 60, 16, dtype=np.int16)
    plt.xticks(10*x, x)
    plt.xlabel('Time [s]')
    plt.ylabel('Power draw [W]')
    plt.legend()
    plt.savefig('fig_data_prep.pdf', dpi=1000, format='pdf')
        
    


fig_data_prep()
# plt.show()