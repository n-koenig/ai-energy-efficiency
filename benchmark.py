import subprocess

reps = 3
experiments = ['keras', 'pytorch']
workloads = [['--', 'bash', '-c', 'source /home/nils/miniconda3/bin/activate tf && python3 tensorflow_mnist.py'], 
            ['--', 'bash', '-c', 'source /home/nils/miniconda3/bin/activate pytorch && python3 pytorch_mnist.py']]
exp_id = 1


with open(f"log_{experiments[exp_id]}.csv", "w") as f:
    if exp_id == 1:
        f.write("epoch,accuracy,loss,val_accuracy,val_loss\n")
    else:
        pass

for i in range(reps):
    with open(f"power_levels_{experiments[exp_id]}_{i}.txt", "w") as f:
        command = ['sudo', 'pinpoint']
        args = ['-r', '1', '--header', '-c', '-a', '8000', '-b', '8000']
        r = subprocess.Popen(command + args + workloads[exp_id], stdin=subprocess.PIPE, stdout=f)
        output, errs = r.communicate()
        # print(output)
