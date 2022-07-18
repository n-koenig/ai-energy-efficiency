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
        self.energy = energy_from_watts(self.watts, offset, interval)
        self.energy_avg = np.average(self.energy, axis=0)
        self.energy_std = 100 * np.std(self.energy, axis=0)/self.energy_avg


def read_watts(path, file_name, reps):
    watts = []
    for i in range(reps):
        watts.append(np.genfromtxt(f"{path}power_levels_{file_name}_{i}.txt", skip_header=1, delimiter=','))    
    return watts

def energy_from_watts(watts, offset, interval):
    energy_sum = np.asarray([np.sum(x[offset+1:-offset]/1000, axis=0) * (interval/1000) for x in watts])
    # energy_sum = np.asarray([np.sum(x[offset+1:-offset], axis=0)/(1000000/interval) for x in watts])
    return energy_sum


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
    plt.savefig('figures/fig_data_prep.pdf', dpi=1000, format='pdf')


def idle_energy():
    sleep = ExperimentData('sleep', 20)
    sleep.set_energy_data('sleep/2/', 200, 100)
    print(sleep.energy_avg)
    print(sleep.energy_std)
    
    

fig_data_prep()
plt.show()