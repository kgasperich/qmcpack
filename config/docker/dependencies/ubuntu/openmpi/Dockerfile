FROM ubuntu:20.04
MAINTAINER William F Godoy williamfgc@yahoo.com

RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get update -y &&\
    apt-get upgrade -y apt-utils

# Dependencies
RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get install gcc g++ \ 
                    clang \
                    clang-format \
                    gcovr \
                    python3 \
                    cmake \
                    ninja-build \
                    libboost-all-dev \
                    git \
                    libopenmpi-dev \
                    libhdf5-openmpi-dev \
                    libhdf5-serial-dev \
                    hdf5-tools \
                    libfftw3-dev \
                    libopenblas-openmp-dev \
                    libxml2-dev \
                    sudo \
                    curl \
                    rsync \
                    wget \
                    software-properties-common \
                    vim \
                    -y

# Python packages for tests
RUN export DEBIAN_FRONTEND=noninteractive &&\
    apt-get install python3-numpy \
                    python3-h5py \
                    python3-pandas \
                    -y

# must add a user different from root 
# to run MPI executables
RUN useradd -ms /bin/bash user
# allow in sudoers to install packages
RUN adduser user sudo
RUN echo "user:user" | chpasswd

USER user
WORKDIR /home/user
