iFROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

ARG OPENCV_VERSION='3.4.11'
ARG GPU_ARCH='6.1'

WORKDIR /opt

# Build tools
RUN apt update && apt install -y sudo tzdata git cmake wget unzip build-essential

# Media I/O:
RUN apt install -y zlib1g-dev libjpeg-dev libwebp-dev \
    libpng-dev libtiff5-dev libopenexr-dev libgdal-dev libgtk2.0-dev

# Video I/O:
RUN apt install -y libdc1394-22-dev libavcodec-dev \
    libavformat-dev libswscale-dev libtheora-dev libvorbis-dev \
    libxvidcore-dev libx264-dev yasm libopencore-amrnb-dev libopencore-amrwb-dev \
    libv4l-dev libxine2-dev libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev ffmpeg

# Parallelism and linear algebra libraries:
RUN apt install -y libtbb-dev libeigen3-dev

# Python:
RUN apt install -y python3-dev python3-tk python3-numpy

# Build OpenCV
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip ${OPENCV_VERSION}.zip && rm ${OPENCV_VERSION}.zip && \
    mv opencv-${OPENCV_VERSION} OpenCV && \
    cd OpenCV && \
    mkdir build && \
    cd build && \
    cmake \
      -D WITH_TBB=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D BUILD_EXAMPLES=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=${GPU_ARCH} \
      -D CUDA_ARCH_PTX=${GPU_ARCH} \
      -D WITH_CUBLAS=ON \
      -D WITH_CUFFT=ON \
      -D WITH_EIGEN=ON \
      -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 \
      .. && \
    make all -j$(nproc) && \
    make install
