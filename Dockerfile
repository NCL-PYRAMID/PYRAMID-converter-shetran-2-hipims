FROM python:3.8-slim

RUN mkdir /app

WORKDIR /app

# TODO: Copy correct files to here
COPY ???
COPY ./requirements.txt ./

COPY ./dl-outputs-sample.zip ./

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENV PLATFORM="docker"

# TODO: Correct Command
#CMD python bbox_to_object.py
