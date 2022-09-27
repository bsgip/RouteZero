# syntax=docker/dockerfile:1

#FROM ubuntu:20.04 as stage_0
FROM python:3.9.5-buster

# Install ssh client and git
RUN apt update
RUN apt install -y openssh-client git

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

WORKDIR /home/
RUN cd /home

# install cbc open source optimisatoin solver for MILP
RUN apt-get update -y
RUN apt-get install -y coinor-cbc

# install packages
COPY ./ ./
RUN pip install numpy
RUN pip install -r requirements.txt
RUN rm -rf /root/.cache/pip

RUN pip install gunicorn

WORKDIR /home/

RUN ls
# gunicorn --workers=10 --threads=1 -b 0.0.0.0:8050 --timeout 600 app:server
CMD ["gunicorn","--workers=10", "--threads=1", "-b 0.0.0.0:8050","-t 600", "app:server"]
