FROM nvcr.io/nvidia/pytorch:23.08-py3

LABEL org.opencontainers.image.authors="yuzhong1214@gmail.com"

ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV SHELL=/bin/bash
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-c"]

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

# Install common tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    htop \
    net-tools \
    tmux \
    vim \
    wget \
    locales \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Install opencv dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
RUN pip3 install \
    numpy \
    matplotlib

ENV PATH=/home/$USERNAME/.local/bin:$PATH
COPY .bashrc /home/$USERNAME/.bashrc

# Set the default user.
USER $USERNAME
CMD ["/bin/bash"]

# Install opencv-python
RUN sudo pip3 uninstall -y opencv \
    && pip3 install opencv-python

# Install custom environment
RUN git clone https://github.com/YuZhong-Chen/.setup_env.git ~/.setup_env && \
    cd ~/.setup_env && ./install.sh

