FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV DEBIAN_FRONTEND=noninteractive

# configure locale
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TZ=Asia/Tokyo
RUN apt-get update && apt-get install -y --no-install-recommends locales tzdata && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen

# configure user and group
RUN apt-get update && apt-get install --no-install-recommends -y sudo && \
    groupadd -g 1000 gwhale && \
    useradd -m -s /bin/bash -u 1000 -g 1000 -G sudo whale && \
    echo whale:password | chpasswd && \
    echo "whale ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER whale

# install basic stuff
RUN sudo apt update && sudo apt install --no-install-recommends -y \
    build-essential curl git less make nano openssh-client ssh software-properties-common vim wget \
    libopencv-dev libcurl4-openssl-dev libssl-dev ffmpeg

WORKDIR /home/whale/tmp

# install go
RUN wget https://go.dev/dl/go1.21.4.linux-amd64.tar.gz && sudo tar -C /usr/local -xzf go1.21.4.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

WORKDIR /home/whale/work
