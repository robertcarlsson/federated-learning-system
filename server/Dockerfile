FROM python:3.5-alpine

RUN apk add docker

RUN apk add \
    python-dev \
    libffi-dev \
    openssl-dev \
    gcc \
    libc-dev \
    make

RUN pip install \
    flask==1.0.2 \
    docker \ 
    docker-compose

WORKDIR /app
COPY . /app

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "/app/app.py" ]