# batch_rl_CQL

Rlkit version with CQL usable with python >=3.8

Rlkit sorce: https://github.com/vitchyr/rlkit

CQL sorce: https://github.com/aviralkumar2907/CQL

# Installation 
To run we used python3.8
Full dependences for the project you can find on install.sh

Run
```sh install.sh```

For mujoco experiments you need to define variables LD_LIBRARY_PATH and WANDB_API_KEY

Example
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia'
export WANDB_API_KEY=''
```
