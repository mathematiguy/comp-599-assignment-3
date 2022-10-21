FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt update

# Create user 'user' to create a home directory
RUN useradd user
RUN mkdir -p /home/user/
RUN chown -R user:user /home/user
ENV HOME /home/user

# Install apt packages
RUN apt update
RUN apt install -y curl wget software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Install python
ENV PYTHON_VERSION 3.9
ENV PYTHON python${PYTHON_VERSION}
RUN apt update
RUN apt install -y ${PYTHON}-dev ${PYTHON}-distutils
RUN rm /usr/bin/python3 && ln -s /usr/bin/${PYTHON} /usr/bin/python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON}

# Set default python version
RUN update-alternatives --install /usr/bin/python python /usr/bin/${PYTHON} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/${PYTHON} 1

# Install pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | ${PYTHON}
RUN update-alternatives --install /usr/local/bin/pip pip /usr/local/bin/pip${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/local/bin/pip3 pip3 /usr/local/bin/pip${PYTHON_VERSION} 1

RUN pip3 install --upgrade pip
COPY requirements.txt /root/requirements.txt
RUN pip3 install -r /root/requirements.txt
