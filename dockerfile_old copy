FROM nvidia/cuda:11.5.2-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y --no-install-recommends software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa -y

RUN apt-get install -y --no-install-recommends python3.9
RUN apt-get install -y --no-install-recommends python3-pip

# Copy the current directory contents into the container at /app
RUN mkdir /app
WORKDIR /app

# Install dependencies from requirements.txt
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN pip install weave

RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y libgl1 
RUN apt-get install -y libglib2.0-0 
RUN rm -rf /var/lib/apt/lists/*
