#!/bin/sh

# This script is set up to work on Ubuntu 16.04 and build tensorflow 1.5
# compiled from source. It'll ask for user input to set up connections with
# your google account (to link with cloud storage) and then it _should_ do
# everything else. I can't promise that future versions of TF/CUDA/CUDNN/Ubuntu
# won't break this and I don't intend to maintain this.

################################################################################
################################################################################
##
## Setting up some variables
##

## NOTE: You will probably need to adjust these yerself

# Home directory
HOME_DIR=~
eval HOME_DIR=$HOME_DIR

# Where should software be installed?
WORK_DIR=$HOME_DIR

# Set this to a google cloud bucket where you will stores 
# the CUDA install files, CUDNN install files, and tensorflow builds.
GCLOUD_BUCKET=gs://coastal-epigram-162302

# Set the version/branch of tensorflow you'd like to build
TF_VERSION=1.6

# Whether to set up a GPU
INSTALL_NVIDIA_UTILS=true
CUDA_DEB_FILE=cuda-repo-ubuntu1604-9-0-local_9.0.176-1_amd64-deb
CUDNN_TGZ_FILE=cudnn-9.0-linux-x64-v7.tgz


# Build TF from source? If false, it'll pull an old version from GCloud
BUILD_NEW_TF=false

# Install some base packages
sudo apt-get install htop git dirmngr aptitude -y


################################################################################
################################################################################
##
## Initialize gcloud utils
##

gcloud init


################################################################################
################################################################################
##
## Cloning Tensorflow source
##

cd $WORK_DIR
git clone https://github.com/tensorflow/tensorflow 
cd tensorflow
git checkout r$TF_VERSION

sudo apt-get install python3-dev python3-setuptools python3-pip python3-wheel python3-numpy -y


################################################################################
################################################################################
##
## If you want GPU-capabilities, this'll install some NVIDIA stuff

if [ "$INSTALL_NVIDIA_UTILS" = true ]; then


    # Set up NVIDIA drivers

    sudo apt-get purge nvidia*
    sudo apt-get autoremove

    sudo apt-get install software-properties-common -y
    sudo add-apt-repository ppa:graphics-drivers -y
    sudo apt-get update
    sudo apt-get install nvidia-384 -y --allow-unauthenticated


    # Install CUDA

    echo 'export CUDA_HOME=/usr/local/cuda-9.0' >> $HOME_DIR/.bashrc
    echo 'export PATH=$PATH:$CUDA_HOME/bin' >> $HOME_DIR/.bashrc
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/lib64' >> $HOME_DIR/.bashrc
    source $HOME_DIR/.bashrc

    cd $WORK_DIR
    wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/$CUDA_DEB_FILE $WORK_DIR

    sudo dpkg -i $CUDA_DEB_FILE
    sudo apt-key add /var/cuda-repo-9-0-local/7fa2af80.pub
    sudo apt-get update -y
    sudo apt-get install cuda-libraries-9-0 -y


    # Install CUDnn

    cd $WORK_DIR
    gsutil cp $GCLOUD_BUCKET/$CUDNN_TGZ_FILE $WORK_DIR
    tar -xvzf $CUDNN_TGZ_FILE

    sudo mkdir $CUDA_HOME/include
    sudo cp cuda/include/cudnn.h $CUDA_HOME/include/cudnn.h

    sudo cp cuda/lib64/* $CUDA_HOME/lib64/
    sudo chmod a+r $CUDA_HOME/include* $CUDA_HOME/lib64/libcudnn*
    sudo ldconfig $CUDA_HOME/lib64

    sudo apt-get install cuda-command-line-tools-9-0 -y
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_HOME/extras/CUPTI/lib64' >> $HOME_DIR/.bashrc
    source $HOME_DIR/.bashrc

fi


################################################################################
################################################################################
##
## Install Bazel
##

sudo apt-get install openjdk-8-jdk -y

echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update -y && sudo apt-get install bazel -y && sudo apt-get upgrade bazel -y



################################################################################
################################################################################
##
## Building tensorflow from source (or copying a pre-built from GCloud)
##

cd $WORK_DIR/tensorflow
bazel shutdown && bazel clean

if [ "$BUILD_NEW_TF" = true ]; then


    # Run config script to tell it what to build
    if [ "$INSTALL_NVIDIA_UTILS" = true ]; then
./configure << EOF
/usr/bin/python3
/usr/local/lib/python3.5/dist-packages
Y
Y
Y
Y
Y
N
N
N
Y
9.0
/usr/local/cuda-9.0
7.1.1
/usr/local/cuda-9.0
3.7
N
/usr/bin/gcc
N
-march=native
N
EOF
    else
./configure << EOF
/usr/bin/python3
/usr/local/lib/python3.5/dist-packages
Y
Y
Y
Y
Y
N
N
N
N
-march=native
N
EOF
    fi

    # Run Bazel to build (this may take 15ish minutes)
    mkdir -p $WORK_DIR/tensorflow/bazel_output

    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
    ./bazel-bin/tensorflow/tools/pip_package/build_pip_package $WORK_DIR/tensorflow/bazel_output

    # This will copy the built .whl files to google cloud so you don't have to rebuild in
    # the future
    gsutil cp $WORK_DIR/tensorflow/bazel_output/tensorflow-$TF_VERSION*.whl $GCLOUD_BUCKET
else
    if [ "$INSTALL_NVIDIA_UTILS" = true ]; then
        WHL_TO_COPY=`gsutil ls -l $GCLOUD_BUCKET/tensorflow-$TF_VERSION*.whl | sort -k2n | tail -n1 | awk 'END {$1=$2=""; sub(/^[     ]+/, ""); print }'`
        gsutil cp $WHL_TO_COPY $WORK_DIR/tensorflow/bazel_output
    fi
fi


################################################################################
################################################################################
##
## Install the compiled tensorflow
##


if [[ "$INSTALL_NVIDIA_UTILS" = true && "$BUILD_NEW_TF" = false ]]; then
    WHL_TO_INSTALL=tensorflow-gpu==$TF_VERSION
else
    WHL_TO_INSTALL=`ls -1t $WORK_DIR/tensorflow/bazel_output/tensorflow-$TF_VERSION*.whl | head -n1`
fi

sudo apt-get install python3-dev python3-setuptools python3-pip python3-wheel python3-numpy -y

sudo -H pip3 install -U pip numpy scipy
sudo -H pip3 install $WHL_TO_INSTALL

# Test tensorflow!

cd $HOME_DIR
python3 - <<-EOF
import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
hello = tf.constant('This is a test. This is only a test. Testy test teste tanuki.')
sess = tf.Session()
print(sess.run(hello))
EOF

# Install some other python packages you may need
sudo -H pip3 install keras pandas ipython jupyter unidecode sklearn
