FROM tensorflow/tensorflow:latest-py3

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install \
    Flask

ENV FLASK_APP /app/app.py

EXPOSE 5000

#ENTRYPOINT [ "sh" ]
ENTRYPOINT [ "python3" ]
CMD [ "/app/app.py" ]