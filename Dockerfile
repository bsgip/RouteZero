# syntax=docker/dockerfile:1

FROM ubuntu:20.04 as stage_0

# Install ssh client and git
RUN apt update
RUN apt install -y openssh-client git

# Download public key for github.com
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

# clone private repositories
RUN --mount=type=ssh git clone git@github.com:bsgip/RouteZero.git

RUN --mount=type=ssh git clone git@github.com:bsgip/dash-blueprint.git


# copy private repos and install on jupyter stack
FROM python:3.9.5-buster as stage_1
USER root
WORKDIR /home/
RUN cd /home

# install cbc open source optimisatoin solver for MILP
RUN apt-get update -y
RUN apt-get install -y coinor-cbc

# install packages
COPY --from=stage_0  RouteZero  RouteZero
COPY --from=stage_0  dash-blueprint dash-blueprint
RUN pip install --no-cache-dir --quiet RouteZero/
RUN pip install dash
RUN pip install --no-cache-dir --quiet dash-blueprint/ && rm -rf dash-blueprint
RUN rm -rf /root/.cache/pip
RUN pip install inflection

RUN cd ./RouteZero
CMD ["python", "RouteZero/app.py"]