FROM ubuntu:bionic

RUN \
  apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
    build-essential \
    cmake \
    libeigen3-dev \
    python3-pip \
    curl \
    wget && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*
  

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda

RUN export PATH="$HOME/miniconda/bin:$PATH" && conda install -c conda-forge pcl -y


COPY . .


RUN export PCL_DIR="$HOME/miniconda/share/pcl-1.9" && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release ../src && \
    cmake --build . && \
    chmod +x 3D_segmentation

RUN pip3 install segyio numpy
RUN pip3 install pymeshfix --no-dependencies

ENTRYPOINT  ["python3", "run_pipeline.py"]
