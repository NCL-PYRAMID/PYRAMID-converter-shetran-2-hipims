FROM python:3.8-slim

RUN mkdir -p /data/inputs
RUN mkdir -p /data/outputs

RUN mkdir /app
WORKDIR /app

COPY ./SHETRAN2HiPIMS.py ./
COPY ./requirements.txt ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PLATFORM="docker"

CMD python SHETRAN2HiPIMS.py
