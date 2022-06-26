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

Example:
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:path/to/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia'
export WANDB_API_KEY=''
```

# Train

Example code of experiments you can fine in path: exaples/
For run example with mujoco run: ```sh run_d4rl_exp.sh```

# Experiment visualization

First, specify variable LD_PRELOAD:

```
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

For visualize police you need run: ```visualize.sh```

or 
```
 python3 scripts/run_policy.py path/to/checkpoints/params.pkl
```