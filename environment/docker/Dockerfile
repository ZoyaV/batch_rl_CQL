FROM nvidia/cuda:11.2.1-cudnn8-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libosmesa6 \
    freeglut3-dev \
    python3.8 \
    libfreetype6 \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    virtualenv \
    wget 
    #software-properties-common \
   # net-tools \
  #  unzip \
  #  vim \
  #  virtualenv \
 #  # wget 
    #xpra \
   # xserver-xorg-dev 
#    && apt-get clean \
 #   && rm -rf /var/lib/apt/lists/

#RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
   # && chmod +x /usr/local/bin/patchelf

RUN apt install python3.8-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.8 get-pip.py

ENV LANG C.UTF-8
WORKDIR /home
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz \
    && tar -xf mujoco210-linux-x86_64.tar.gz -C /root/.mujoco \
    && rm mujoco210-linux-x86_64.tar.gz
#COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

#COPY vendor/Xdummy /usr/local/bin/Xdummy
#RUN chmod +x /usr/local/bin/Xdummy

# Workaround for https://bugs.launchpad.net/ubuntu/+source/nvidia-graphics-drivers-375/+bug/1674677
#COPY ./vendor/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

#RUN apt-get update && apt-get install -y libav-tools

# For some reason this works despite an error showing up...
#RUN DEBIAN_FRONTEND=noninteractive apt-get -qy install nvidia-384; exit 0
#ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/usr/lib/nvidia-384


# For atari-py
RUN apt-get install -y zlib1g-dev swig cmake
RUN apt-get install libjpeg-dev zlib1g-dev -y
# Previous versions installed from a requirements.txt, but direct pip
# install seems cleaner
RUN pip3 install Pillow>=4.0.0
RUN pip3 install torch torchvision torchaudio
RUN git clone https://github.com/ZoyaV/CQL.git
RUN cd CQL/d4rl && pip3 install -e.


# fixing ruamel.yaml
RUN pip3 install --upgrade pip
RUN pip3 install python-dateutil
RUN pip3 install matplotlib
RUN pip3 install gtimer==1.0.0b5
RUN pip3 install h5py
RUN pip install glfw>=1.4.0
RUN pip install numpy>=1.11
RUN pip install Cython>=0.27.2
RUN pip install imageio>=2.1.2
RUN pip install cffi>=1.10
RUN pip install imagehash>=3.4
RUN pip install ipdb
RUN pip install pycparser>=2.17.0
RUN pip install pytest>=3.0.5
RUN pip install pytest-instafail==0.3.0
RUN pip install scipy>=0.18.0
RUN pip install sphinx
RUN pip install sphinx_rtd_theme
RUN pip install numpydoc
RUN pip install cloudpickle==0.5.2
RUN pip install cached-property==1.3.1
RUN pip install gitpython==2.1.7
RUN pip install awscli==1.11.179
RUN pip install boto3==1.4.8
RUN pip install ray
RUN pip install path.py==10.3.1
RUN pip install joblib==0.9.4
RUN pip install opencv-python
RUN pip install sk-video==1.1.10
RUN pip install git+https://github.com/vitchyr/multiworld.git
RUN pip install mujoco-py==0.5.7
RUN pip3 install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
RUN pip install /home/CQL/d4rl
#COPY CQL/d4rl 
#RUN pip install -e .
#RUN git clone https://github.com/ZoyaV/CQL.git
#RUN pip install rlkit
#RUN cd CQL/d4rl && pip install -e.

