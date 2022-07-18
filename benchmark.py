import os
import subprocess

def run_experiment(reps, out_path, exp_name, workload):
    for i in range(reps):
        with open(out_path + f"power_levels_{exp_name}_{i}.txt", "w") as f:
            command = ['sudo', 'pinpoint']
            command += ['-r', '1']
            command += ['-a', '20000']
            command += ['-b', '20000']
            command += ['-c']
            command += ['-i', '100']
            command += ['--header']
            command += ['--']
            command += workload
            r = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=f)
            output, errs = r.communicate()
            # print(output)


def data_amount_exp():
    exp_id = 0
    for i in range(10):
        workload = []
        workload += workload_envs[exp_id]
        workload[-1] += script_paths[exp_id]
        workload[-1] += f' {(i+1)*10}'
        exp_name = experiments[0] + f'_{(i+1)*10}'
        out_path = output_paths[1]
        os.makedirs(out_path, exist_ok=True)
        run_experiment(20, out_path, exp_name, workload)


def compare_exp():
    workload = []
    workload += workload_envs[0]
    workload[-1] += script_paths[0]
    exp_name = experiments[0]
    out_path = output_paths[1]
    os.makedirs(out_path, exist_ok=True)
    run_experiment(20, out_path, exp_name, workload)
    workload = []
    workload += workload_envs[1]
    workload[-1] += script_paths[1]
    exp_name = experiments[1]
    out_path = output_paths[1]
    run_experiment(20, out_path, exp_name, workload)



experiments = ['keras', 'pytorch', 'sleep', 'stress', 'sleep_summary', 'sleep_summary_r']
workload_envs = [['bash',  '-c', 'source /home/nils/miniconda3/bin/activate tf && python3 '],
                ['bash', '-c', 'source /home/nils/miniconda3/bin/activate pytorch && python3 ']]
script_paths = ['MNIST_CNN/keras_mnist.py', 'MNIST_CNN/pytorch_mnist.py']
output_paths = ["dump/", "MNIST_CNN/6/", 'sleep/2/', 'stress/', 'pinpoint_testing/', 'dump2/']

exp_id = 3
reps = 20
out_path = output_paths[2]
exp_name = experiments[exp_id]
# workload = workload_envs[exp_id]
# workload[-1] += script_paths[exp_id]
workload = ['sleep', '10']
# workload = ['stress', '--cpu',  '8', '--io', '4', '--vm', '20', '--vm-bytes', '128M', '--timeout', '10s', '-q']

os.makedirs(out_path, exist_ok=True)
# with open(f"log_{exp_name}.csv", "w") as f:
#     if exp_id == 1:
#         f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")
#     else:
#         pass

# data_amount_exp()
compare_exp()
# run_experiment(reps, out_path, exp_name, workload)

