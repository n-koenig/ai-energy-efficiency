# ai-energy-efficiency

## next steps:
- compute energy from raw data
- collect and compute average accuracy and energy
- further compute standard deviaton and what not for accuracy and energy
- look into pytorch lightning/benchmarking
- look into wisemat model

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

## execution
- conda activate tf; python3 tensorflow_mnist.py
- conda activate pytorch; python3 pytorch_mnist.py
