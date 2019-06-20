FROM tensorflow/tensorflow:latest-py3

COPY . /app

RUN pip install --upgrade pip
RUN pip install \
    Flask

ENV FLASK_APP /app/app.py

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "/app/app.py" ]