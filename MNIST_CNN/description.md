# Simple MNIST CNN
- cnn, implemented in pytorch and keras, trained on mnist

## hyperparameters
- 12 epochs, batch size 128


## pinpoint 
- 20 runs each, through benchmark.py
- -i 50, -a 8000, -b 8000


## results/learnings:
- pytorch much more energy
- pytorch better accuracy
- keras better (linear) efficiency

## notes
- 1: measurements after PC restart
- 2: measurements after PC was already used
- no programs opened, no specific power settings used



- 4, 5 both used new benchmark script which executes all consecutively and also random-seed-setting and interval 100
- 4 used default (lr=0.001 i think), 5 used lr=0.01
