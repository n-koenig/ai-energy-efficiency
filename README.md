# ai-energy-efficiency

## next steps:
- measure sleep/idle consumption
- check pytorch model and keras model
    - maybe check for the random seed stuff
- test different pinpoint execution styles (and maybe validate values against nvidia-smi)
- check energy calculation (sampling interval, correct integration, subtract idle power)


- write script to run all sub-experiments directly consecutively


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


- currently paths not working for accuracy logs yet