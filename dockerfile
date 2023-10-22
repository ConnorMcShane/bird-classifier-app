FROM tensorflow/tensorflow:2.14.0-gpu

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get upgrade -y

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt