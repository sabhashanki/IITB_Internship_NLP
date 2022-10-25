FROM python:3.10.7
ARG port
USER root
COPY . /app
WORKDIR /app
ENV PORT=$port
RUN apt-get update && apt-get install -y --no-install-recommends apt-utils \
    && apt-get -y install curl \
    && apt-get install libgomp1
RUN chgrp -R 0 /app \
    && chmod -R g=u /app \
    && pip install pip --upgrade \
    && pip install -r requirements.txt
EXPOSE $PORT
CMD gunicorn app:server --bind 0.0.0.0:$PORT --preload