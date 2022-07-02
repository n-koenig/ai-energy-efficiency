import subprocess

reps = 1
experiments = ['keras', 'pytorch']
workloads = [['--', 'bash', '-c', 'source /home/nils/miniconda3/bin/activate tf && python3 tensorflow_mnist.py'], 
            ['--', 'bash', '-c', 'source /home/nils/miniconda3/bin/activate pytorch && python3 pytorch_mnist.py']]


# with open("accuracy.txt", "w") as f:
# 	pass

exp_id = 0
for i in range(10):
    with open(f"power_levels_{experiments[exp_id]}_{i}.txt", "w") as f:
        command = ['sudo', 'pinpoint']
        args = ['-r', str(reps), '--header', '-c', '-a', '8000', '-b', '8000']
        r = subprocess.Popen(command + args + workloads[exp_id], stdin=subprocess.PIPE, stdout=f)
        output, errs = r.communicate()
        # print(output)
