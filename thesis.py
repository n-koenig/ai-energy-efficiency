import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class ExperimentData:
    def __init__(self, name, reps, offset, interval):
        self.name = name
        self.reps = reps
        self.offset = int(offset/interval)
        self.interval = interval
        self.watts = []
        self.energy = []
        self.energy_avg = []
        self.energy_std = []
        self.acc_avg = 0
        self.acc_std = 0
        self.efficiency = 0


    def set_energy_data(self, path):
        self.watts = read_watts(path, self.name, self.reps)
        self.energy = energy_from_watts(self.watts, self.offset, self.interval)
        self.energy_avg = np.average(self.energy, axis=0)
        self.energy_std = 100 * np.std(self.energy, axis=0)/self.energy_avg

    def set_acc_data(self, path, epochs):
        self.accs = read_accs(path, self.name, self.reps, epochs)
        self.acc_avg = np.average(self.accs)
        self.acc_std = 100*np.std(self.accs)/self.acc_avg


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


def energy_from_watts(watts, offset, interval):
    energy_sum = np.asarray([np.sum(x[offset+1:-offset], axis=0) * (interval/1000) for x in watts])
    # energy_sum = np.asarray([np.sum(x[offset+1:-offset], axis=0)/(1000000/interval) for x in watts])
    return energy_sum


def tolerant_mean(arrs):
    # https://stackoverflow.com/a/59281468
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)


def fig_data_prep():
    data = ExperimentData('keras_10', 3, 20000, 100)
    data.watts = read_watts('MNIST_CNN/5/', data.name, data.reps)
    for x in data.watts:
        x[:, 0] -= 16000
        x[:, 0] /= 1000
        x[:, 2] -= 1000
        x[:, 2] /= 1000
    data.energy = energy_from_watts(data.watts, data.offset, data.interval)
    data.energy_avg = np.average(data.energy, axis=0)

    plt.figure()
    plt.title('GPU Power Draw')
    for i in range(3):
        plt.plot(data.watts[i][:, 0]/1000, label=f'Run {i}')
    plt.axvline(x=200, color='r')
    plt.axvline(x=np.average([len(x) for x in data.watts])-200, color='r')
    x = np.linspace(0, 60, 7, dtype=np.int16)
    plt.xticks(10*x, x)
    plt.xlabel('Time [s]')
    plt.ylabel('Power draw [W]')
    plt.legend()
    plt.savefig('figures/preprocessing.pdf', dpi=1000, format='pdf')


def idle_power():
    sleep = ExperimentData('sleep', 20, 20000, 100)
    sleep.set_energy_data('sleep/2/')
    print(sleep.energy_avg)
    print(sleep.energy_std)


def eval_compare():
    path = 'MNIST_CNN/2/'
    titles = ['Keras GPU', 'PyTorch GPU', 'Keras CPU', 'PyTorch CPU']
    reps = 20
    offset = 8000
    interval = 50
    
    keras = ExperimentData('keras', reps, offset, interval)
    keras.watts = read_watts(path, keras.name, keras.reps)
    for x in keras.watts:
        x[:, 0] -= 16000
        x[:, 0] /= 1000
        x[:, 2] -= 1000
        x[:, 2] /= 1000
    keras.energy = energy_from_watts(keras.watts, keras.offset, keras.interval)
    keras.energy_avg = np.average(keras.energy, axis=0)
    keras.energy_std = 100 * np.std(keras.energy, axis=0)/keras.energy_avg
    keras.set_acc_data(path, 12)

    pytorch = ExperimentData('pytorch', 20, 8000, 50)
    pytorch.watts = read_watts(path, pytorch.name, pytorch.reps)
    for x in pytorch.watts:
        x[:, 0] -= 16000
        x[:, 0] /= 1000
        x[:, 2] -= 1000
        x[:, 2] /= 1000
    pytorch.energy = energy_from_watts(pytorch.watts, pytorch.offset, pytorch.interval)
    pytorch.energy_avg = np.average(pytorch.energy, axis=0)
    pytorch.energy_std = 100 * np.std(pytorch.energy, axis=0)/pytorch.energy_avg
    pytorch.set_acc_data(path, 12)


    print(f'Keras Energy: GPU: {keras.energy_avg[0]:,.2f} J ({keras.energy_std[0]:,.2f}%), CPU: {keras.energy_avg[2]:,.2f} J ({keras.energy_std[2]:,.2f}%), Total: {keras.energy_avg[0] + keras.energy_avg[2]:,.2f} J')
    print(f'PyTorch Energy: GPU: {pytorch.energy_avg[0]:,.2f} J ({pytorch.energy_std[0]:,.2f}%), CPU: {pytorch.energy_avg[2]:,.2f} J ({pytorch.energy_std[2]:,.2f}%), Total: {pytorch.energy_avg[0] + pytorch.energy_avg[2]:,.2f} J')
    print(keras.accs, 100*keras.acc_avg, keras.acc_std)
    print(pytorch.accs, 100*pytorch.acc_avg, pytorch.acc_std)
    print(f'Keras & {keras.energy_avg[0] + keras.energy_avg[2]:,.2f} & {100*keras.acc_avg:.2f} \\\ \nPyTorch & {pytorch.energy_avg[0] + pytorch.energy_avg[2]:,.2f} & {100*pytorch.acc_avg:.2f}')

    fig, axs = plt.subplots(2, 2, sharex='col', sharey='all')
    fig.suptitle('Average Power Draw')
    l = list()
    
    arr = []
    arr.append([x[:, 0] for x in keras.watts])
    arr.append([x[:, 0] for x in pytorch.watts])
    arr.append([x[:, 2] for x in keras.watts])
    arr.append([x[:, 2] for x in pytorch.watts])
    for i in range(4):
        y, error = tolerant_mean(arr[i])
        x = np.linspace(0, y.shape[0]-1, y.shape[0])
        index = [int(x) for x in f"{i:02b}"]
        l.append(axs[index[0], index[1]].plot(y, label='average'))
        axs[index[0], index[1]].fill_between(x, y-error, y+error, color='green', alpha=0.2, label='standard\ndeviation')
        axs[index[0], index[1]].set_title(titles[i])
        axs[index[0], index[1]].axvline(keras.offset, color='r')
        axs[index[0], index[1]].axvline(len(y)-keras.offset, color='r')
        if index[0]: 
            axs[index[0], index[1]].set_xlabel('Time [s]')
        if not index[1]: 
            axs[index[0], index[1]].set_ylabel('Power Draw [W]')
            x = np.linspace(0, 100, 6, dtype=np.int16)
            axs[index[0], index[1]].set_xticks(20*x, x)
        else:
            x = np.linspace(0, 140, 8, dtype=np.int16)
            axs[index[0], index[1]].set_xticks(20*x, x)
    for a in fig.axes:
        a.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        a.tick_params(axis='y', which='both', bottom=True, top=False, labelleft=True)
    fig.tight_layout()
    plt.savefig('figures/compare_watts.pdf', dpi=1000, format='pdf')


    # fig, axs = plt.subplots(1, 2, sharex='all', sharey='all')
    # fig.suptitle('Average Power Draw')
    # l = list()
    
    # arr = []
    # arr.append([x[:, 0] for x in keras.watts])
    # arr.append([x[:, 2] for x in keras.watts])
    # arr.append([x[:, 0] for x in pytorch.watts])
    # arr.append([x[:, 2] for x in pytorch.watts])
    # for i in range(2):
    #     index = [int(x) for x in f"{i:02b}"]
    #     for j in range(2):
    #         y, error = tolerant_mean(arr[(i*2)+j])
    #         x = np.linspace(0, y.shape[0]-1, y.shape[0])
    #         l.append(axs[i].plot(y, label='average'))
    #         axs[i].fill_between(x, y-error, y+error, color='green', alpha=0.2, label='standard\ndeviation')
    #         axs[i].set_title(titles[i])
    #         axs[i].axvline(keras.offset)
    #         axs[i].axvline(len(y)-keras.offset)
    #         axs[i].set_xlabel('Time [s]')
    #         axs[i].set_ylabel('Power Draw [W]')
    #     for a in fig.axes:
    #         a.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
    #         a.tick_params(axis='y', which='both', bottom=True, top=False, labelleft=True)
    # # fig.text(0.5, 0.1, 'Time [s]', ha='center', va='center')
    # # fig.text(0.06, 0.5, 'Power Draw [W]', ha='center', va='center', rotation='vertical')
    # # fig.legend(l, labels=['average', 'standard\ndeviation'], loc="right")
    # # axs[0, 0].legend()
    # fig.tight_layout()
    

def eval_data():
    path = 'MNIST_CNN/4/'
    reps = 20
    offset = 20000
    interval = 100
    
    data = []
    energy_gpu = []
    energy_cpu = []
    energy_total = []
    acc = []
    for i in range(10):
        data.append(ExperimentData(f'keras_{(i+1)*10}', reps, offset, interval))

        data[i].watts = read_watts(path, data[i].name, data[i].reps)
        for x in data[i].watts:
            x[:, 0] -= 16000
            x[:, 0] /= 1000
            x[:, 2] -= 1000
            x[:, 2] /= 1000
        data[i].energy = energy_from_watts(data[i].watts, data[i].offset, data[i].interval)
        data[i].energy_avg = np.average(data[i].energy, axis=0)
        data[i].energy_std = 100 * np.std(data[i].energy, axis=0)/data[i].energy_avg
        data[i].set_acc_data(path, 12)

        energy_gpu.append(data[i].energy_avg[0])
        energy_cpu.append(data[i].energy_avg[2])
        energy_total.append(data[i].energy_avg[0] + data[i].energy_avg[2])
        acc.append(100*data[i].acc_avg)

    table = ''
    for i in range(10):
        table += f'{(i+1)*6000} & {energy_total[i]:,.2f} & {acc[i]:.2f} \\\ \n'
    print(table)

    plt.figure()
    plt.title('Average Energy Consumption')
    labels = [f'{i*6}' for i in range(1, 11, 1)]
    plt.xlabel('Number of training samples [n*1000]')
    plt.ylabel('Energy consumption [J]')
    plt.bar(labels, energy_gpu, label="GPU")
    plt.bar(labels, energy_cpu, bottom=energy_gpu, label="CPU")
    plt.legend()
    # plt.savefig('figures/data_energy.pdf', format='pdf')

    plt.figure()
    def func(x, a, b):
        return a * np.log(x) + b
    popt, pcov = optimize.curve_fit(func, energy_total, acc)
    plt.scatter(energy_total, acc)
    plt.plot(energy_total, func(energy_total, *popt))
    plt.title('Average Accuracy')
    plt.xlabel('Energy consumption [J]')
    plt.ylabel('Accuracy [%]')
    plt.ylim((0, 100))
    # plt.savefig('figures/data_acc.pdf', format='pdf')

    # plt.figure()
    # plt.title('GPU Power Draw')
    # plt.plot(data[0].watts[0][:, 0]/1000, label=f'Run {0}')
    # plt.plot(data[3].watts[0][:, 0]/1000, label=f'Run {3}')
    # plt.plot(data[6].watts[0][:, 0]/1000, label=f'Run {6}')
    # plt.plot(data[9].watts[0][:, 0]/1000, label=f'Run {9}')
    # # plt.axvline(x=200, color='r')
    # # plt.axvline(x=np.average([len(x) for x in data.watts])-200, color='r')
    # x = np.linspace(0, 60, 7, dtype=np.int16)
    # plt.xticks(10*x, x)
    # plt.xlabel('Time [s]')
    # plt.ylabel('Power draw [W]')
    # plt.legend()
    # plt.savefig('figures/data_power.pdf', dpi=1000, format='pdf')


    for i in range(10):
        plt.figure()
        # fig, axs = plt.subplots(2, 2, sharex='col', sharey='all')
        # fig.suptitle('Average Power Draw')
        # l = list()
        arr = []
        arr.append([x[:, 0] for x in data[i].watts])
        arr.append([x[:, 2] for x in data[i].watts])
        y, error = tolerant_mean(arr[0])
        x = np.linspace(0, y.shape[0]-1, y.shape[0])
        plt.plot(y, label='GPU')
        plt.fill_between(x, y-error, y+error, color='green', alpha=0.2)
        plt.title('Average GPU and CPU Power Draw')
        plt.axvline(data[i].offset, color='r')
        plt.axvline(len(y)-data[i].offset, color='r')
        y, error = tolerant_mean(arr[1])
        x = np.linspace(0, y.shape[0]-1, y.shape[0])
        plt.plot(y, label='CPU')
        plt.fill_between(x, y-error, y+error, color='green', alpha=0.2, label='standard\ndeviation')
        plt.savefig(f'figures/data_{(i+1)*10}_watts.pdf', dpi=1000, format='pdf')
        if i == 0:
            plt.legend()
            plt.savefig(f'figures/data_{(i+1)*10}_watts_legend.pdf', dpi=1000, format='pdf')
        if i == 7:
            plt.legend()
            plt.savefig(f'figures/data_{(i+1)*10}_watts_legend.pdf', dpi=1000, format='pdf')
            
        
        
        
        
        # for i in range(4):
        #     y, error = tolerant_mean(arr[i])
        #     x = np.linspace(0, y.shape[0]-1, y.shape[0])
        #     index = [int(x) for x in f"{i:02b}"]
        #     l.append(axs[index[0], index[1]].plot(y, label='average'))
        #     axs[index[0], index[1]].fill_between(x, y-error, y+error, color='green', alpha=0.2, label='standard\ndeviation')
        #     axs[index[0], index[1]].set_title(titles[i])
        #     axs[index[0], index[1]].axvline(keras.offset, color='r')
        #     axs[index[0], index[1]].axvline(len(y)-keras.offset, color='r')
        #     if index[0]: 
        #         axs[index[0], index[1]].set_xlabel('Time [s]')
        #     if not index[1]: 
        #         axs[index[0], index[1]].set_ylabel('Power Draw [W]')
        #         x = np.linspace(0, 100, 6, dtype=np.int16)
        #         axs[index[0], index[1]].set_xticks(20*x, x)
        #     else:
        #         x = np.linspace(0, 140, 8, dtype=np.int16)
        #         axs[index[0], index[1]].set_xticks(20*x, x)
        # for a in fig.axes:
        #     a.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        #     a.tick_params(axis='y', which='both', bottom=True, top=False, labelleft=True)
        # fig.tight_layout()
        # plt.savefig(f'figures/data_{(i+1)*10}_watts.pdf', dpi=1000, format='pdf')


# fig_data_prep()
# eval_compare()
eval_data()
plt.show()