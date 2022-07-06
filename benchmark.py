import os
import subprocess

experiments = ['keras_40', 'pytorch']
workload_envs = [['bash',  '-c', 'source /home/nils/miniconda3/bin/activate tf && python3 '],
            ['bash', '-c', 'source /home/nils/miniconda3/bin/activate pytorch && python3 ']]
script_paths = ['MNIST_CNN/keras_mnist.py', 'MNIST_CNN/pytorch_mnist.py']
output_paths = ["dump/", "MNIST_CNN/3/"]

exp_id = 0
reps = 20
out_path = output_paths[1]
exp_name = experiments[exp_id]
workload = workload_envs[exp_id]
workload[-1] += script_paths[exp_id]

os.makedirs(out_path, exist_ok=True)
with open(f"log_{exp_name}.csv", "w") as f:
    if exp_id == 1:
        f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")
    else:
        pass

for i in range(reps):
    with open(out_path + f"power_levels_{exp_name}_{i}.txt", "w") as f:
        command = ['sudo', 'pinpoint']
        command += ['-r', '1']
        command += ['-a', '20000']
        command += ['-b', '20000']
        command += ['-c']
        command += ['--header']
        command += ['--']
        command += workload
        r = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=f)
        output, errs = r.communicate()
        # print(output)
