FROM nvidia/cuda:9.0-base-ubuntu16.04
RUN apt-get update && apt-get install -y --no-install-recommends \
	curl sudo build-essential git \
 && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /App
WORKDIR /App


# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash chenpingjun \
 && chown -R chenpingjun:chenpingjun /App
RUN echo "chenpingjun ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-chenpingjun
USER chenpingjun
ENV HOME=/home/chenpingjun
RUN chmod 777 /home/chenpingjun


# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/chenpingjun/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false


# Create a Python 3.6 environment
RUN /home/chenpingjun/miniconda/bin/conda install conda-build \
 && /home/chenpingjun/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/chenpingjun/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/chenpingjun/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH


# CUDA 9.0-specific steps
RUN conda install -y -c pytorch cuda90=1.0 magma-cuda90=2.4.0 \
    "pytorch=1.0.0=py3.6_cuda9.0.176_cudnn7.4.1_1" \
    torchvision=0.2.1 \
 && conda clean -ya


# PIP packages
RUN pip install --upgrade pip && \
    pip install --upgrade setuptools
COPY requirements.txt /tmp/
RUN pip install --upgrade --ignore-installed -r /tmp/requirements.txt
