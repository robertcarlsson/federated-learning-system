FROM ubuntu:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.5 \
    python3-pip


RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

RUN pip install --upgrade pip 

RUN pip install \
    Flask==1.0.2
#&& \
#    hash pip && \
#    pip install \
#    Flask
#    pip install -Iv --no-cache-dir \
#    Flask==1.0.2

#RUN pip install Flask

#COPY . /appw

#ADD ./federation /app/federation
#ADD ./device /app/device

COPY . /app

EXPOSE 5000
ENTRYPOINT [ "python" ]
CMD [ "/app/app.py" ]