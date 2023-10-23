FROM tensorflow/tensorflow:2.13.0-gpu

# Install any needed packages specified in requirements.txt
RUN apt-get update -y
RUN apt-get upgrade -y

# Copy the current directory contents into the container at /app
RUN mkdir /app
WORKDIR /app
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install --upgrade requests
RUN pip install .

RUN apt-get install -y libgl1 
RUN apt-get install -y libglib2.0-0

# Make port 8000 available to the world outside this container
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]