FROM tensorflow/tensorflow:2.13.0-gpu

# Install any needed packages specified in requirements.txt
RUN apt-get update -y
RUN apt-get upgrade -y

# Copy the current directory contents into the container at /app
RUN mkdir /app
WORKDIR /app

RUN apt-get install -y libgl1 
RUN apt-get install -y libglib2.0-0 
RUN rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements.txt
COPY requirements.txt /app
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
