Learning Based Control
=====

## Overview

This repository is an extension of the code found at `https://github.com/bpriviere/decision_making`.

B. Riviere, W. Hönig, M. Anderson, S-J. Chung. "Neural Tree Expansion for Multi-Robot Planning in Non-Cooperative Environments" in IEEE Robotics and Automation Letters (RA-L) June 2021. 


## Dependencies:

This project should be installed into a virtual environment using the following command:

```
pip install -e .
```

If this fails, remove the offending entry from `requirements.txt` and try again.

Due to `pytorch` having further dependencies and run configurations, that package is best installed individually. For instance, if running on Windows, , pip, and cuda 11.3, the command is as follows:

```
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio===0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Refer to the `pytorch` documentation for more information regarind install configurations. 

```
https://pytorch.org/get-started/locally/
```

## Compiling:
from `~/code/cpp`:
```
mkdir build
cd build
cmake -DPYTHON_EXECUTABLE=$(which python) -DCMAKE_BUILD_TYPE=Release ..
make
```

## Script Examples:  
Run individual problems and solvers from `~\code` by modifying `code/param.py` and then:
```
python run.py
```
Run batch examples from `~\code\tests`: 
```
python regression.py
```
Run waypoint planning from `~\code\tests`: 
```
python waypoint_planning.py
```
Train neural networks by modifying parameters in `code/train.py` then, from `~\code`:
```
python train.py
```

## Error Messages

If you get an error message such as: 

```No such file or directory: '/home/ben/projects/decision_making/saved/example9/model_value_l0.pt```

it means that the solver tried to query a neural network oracle that does not exist. You can either disable neural network search or create a new model. 

To disable the neural network search, change `oracles_on` in `param.py` to `oracles_on = False`

To create a model, run `python train.py`. After training, you can query the newly created model (in `../current/models/`) by changing the `dirname` parameter in `param.py` to the corresponding location. 