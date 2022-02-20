FROM tensorflow/tensorflow:latest-py3

RUN pip install --upgrade pip
RUN pip install \
    requests

COPY . /app

ENTRYPOINT [ "python3" ]
# Change the ip adress to the device running docker
CMD [ "/app/app.py" , "192.168.1.105"]