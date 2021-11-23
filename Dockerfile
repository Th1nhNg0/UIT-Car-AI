# set base image (host OS)
FROM python:3.9

RUN apt-get update ##[edited]
RUN apt-get install ffmpeg libsm6 libxext6  -y

# set the working directory in the container
WORKDIR /code

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY model/ model/
COPY client.py .
COPY model.py .

# command to run on container start
CMD [ "python", "./client.py" ]