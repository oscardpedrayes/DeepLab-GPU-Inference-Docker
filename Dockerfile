FROM tensorflow/tensorflow:latest-gpu

ADD . /code
WORKDIR /code

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]