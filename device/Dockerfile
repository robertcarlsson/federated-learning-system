FROM tensorflow/tensorflow:latest-py3

RUN pip install --upgrade pip
RUN pip install \
    requests

COPY . /app

ENTRYPOINT [ "python" ]
CMD [ "/app/app.py" ]