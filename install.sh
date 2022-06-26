pip install --upgrade pip setuptools wheel
pip install gtimer==1.0.0b5
pip install dm-control

sudo apt-get install libhdf5-dev
sudo apt-get install libosmesa6-dev
pip install --upgrade "protobuf<=3.20.1"

pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
pip install patchelf

pip install -e.

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/zoya/.mujoco/mujoco210/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

#wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
#tar -xvzf mujoco210-linux-x86_64.tar.gz -C .mujoco
