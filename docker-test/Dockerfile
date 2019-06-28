FROM python:3.5-alpine

WORKDIR /app

RUN apk add docker

RUN pip install \
    flask==1.0.2 \
    docker

COPY . .

ENTRYPOINT [ "sh" ]