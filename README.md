# Setting up Deep Learning Machine with Python3

## Install [Cuda 8.0](https://developer.nvidia.com/cuda-downloads)
    $ sudo dpkg -i cuda-repo-ubuntu1604-8-0-local-ga2_8.0.61-1_amd64.deb
    $ sudo apt-get update
    $ sudo apt-get install cuda
    $ echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    $ echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    $ source ~/.bashrc

## Check Driver Installation
    $ nvidia-smi
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 375.39                 Driver Version: 375.39                    |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 970     Off  | 0000:01:00.0      On |                  N/A |
    |  0%   44C    P5    21W / 163W |    413MiB /  4034MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+
                                                                               
    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID  Type  Process name                               Usage      |
    |=============================================================================|
    |    0      2964    G   /usr/lib/xorg/Xorg                             293MiB |
    |    0      3796    G   compiz                                         116MiB |
    |    0      4342    G   /usr/lib/firefox/firefox                         2MiB |
    +-----------------------------------------------------------------------------+ 

## Check Cuda Installation
    $ nvcc -V
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2016 NVIDIA Corporation
    Built on Tue_Jan_10_13:22:03_CST_2017
    Cuda compilation tools, release 8.0, V8.0.61

## Install [Cudnn 5.1](https://developer.nvidia.com/rdp/cudnn-download)
    $ cd ~/Downloads/
    $ tar xvf cudnn*.tgz
    $ cd cuda
    $ sudo cp */*.h /usr/local/cuda/include/
    $ sudo cp */libcudnn* /usr/local/cuda/lib64/
    $ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
    
## Install [Tensorflow 1.0](https://www.tensorflow.org/install/install_linux#InstallingNativePip)
    $ sudo pip3 install tensorflow-gpu

## Check TF installation
    $ python3
    >>> import tensorflow
    I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
    I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
    I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
    I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
    I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally

## Install [Keras 1.2.2](https://keras.io/#installation)
    $ sudo pip3 install keras==1.2.2
    
## Check Keras installation
    >>> import keras
    Using TensorFlow backend.
    
## Install Opencv
    $ sudo pip3 install opencv-python
    
## Install [OpenAI Gym](https://github.com/openai/gym)
    $ sudo pip3 install gym
    
## Install [Dlib](https://github.com/davisking/dlib)
    $ sudo apt-get install cmake
    $ sudo apt-get install libboost-all-dev
    $ git clone https://github.com/davisking/dlib.git
    $ cd dlib
    $ sudo python3 setup.py install --yes USE_AVX_INSTRUCTIONS
    
## Install Additional Python3 Packages
    $ sudo pip3 install scipy sklearn pandas matplotlib scikit-image pillow flask-socketio eventlet jupyter h5py moviepy

## Scipy Tutorials
[computational statistics in python](http://people.duke.edu/~ccc14/sta-663/usingnumpysolutions.html)

[scipy-lectures](http://www.scipy-lectures.org/intro/numpy/advanced_operations.html)
