# syntax=docker/dockerfile:1

#FROM ubuntu:20.04 as stage_0
FROM python:3.9.5-buster

# Install ssh client and git
RUN apt update
RUN apt install -y openssh-client git

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# clone private repositories
#RUN --mount=type=ssh git clone -b deploy git@github.com:bsgip/RouteZero.git

#RUN --mount=type=ssh git clone git@github.com:bsgip/dash-blueprint.git


# copy private repos
#FROM python:3.9.5-buster as stage_1
USER root
WORKDIR /home/
RUN cd /home

# install cbc open source optimisatoin solver for MILP
RUN apt-get update -y
RUN apt-get install -y coinor-cbc

# install packages
#COPY --from=stage_0  RouteZero  RouteZero
#COPY --from=stage_0  dash-blueprint dash-blueprint
RUN --mount=type=ssh git clone -b docker git@github.com:bsgip/RouteZero.git
RUN cd ./RouteZero && pip install . && cd ../
RUN --mount=type=ssh pip install -e git+ssh://git@github.com/bsgip/dash-blueprint.git@master#egg=dash_blueprint
RUN rm -rf /root/.cache/pip

RUN pip install gunicorn

WORKDIR /home/RouteZero/

RUN ls
# gunicorn --workers=10 --threads=1 -b 0.0.0.0:8050 --timeout 600 app:server
CMD ["gunicorn","--workers=10", "--threads=1", "-b 0.0.0.0:8050","-t 600", "app:server"]
# "--chdir ./RouteZero"