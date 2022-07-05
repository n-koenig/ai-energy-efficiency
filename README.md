# ai-energy-efficiency

## next steps:
- compute energy from raw data
- collect and compute average accuracy and energy
- further compute standard deviaton and what not for accuracy and energy

---

- write functions for different calculations/visualizations from raw data
- look into pytorch lightning/benchmarking
- look into wisemat model

## for getemed presentation:
- mnist models with keras/pytorch, 12 epochs*20 runs
- plot average energy for each device+start/stop lines + standard deviation range
- plot avg energy vs avg accuracy for each


## conda
- conda install -n <name> <package>
- conda remove -n <name> <package>
- conda env list
- conda env remove -n <name>
- conda create -n <name> python=3.9

- for tensorflow: https://www.tensorflow.org/install/pip working, but replace 11.2 with 11.7
- for pytorch: `pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117`

## nvidia-smi
- nvidia-smi -i 0 -q -d POWER


## notes
- make log files that are opened inside python scripts 666 (r+w for everyone)
- two consecutive GPU watt values are always exactly the same, probably can not be sampled that often

## execution
- conda activate tf; python3 tensorflow_mnist.py
- conda activate pytorch; python3 pytorch_mnist.py
