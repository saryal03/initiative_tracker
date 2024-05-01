FROM python:3.12-slim
WORKDIR /initiativetracker-main
COPY bert_model/ /initiativetracker-main/bert_model/
COPY bert.py /initiativetracker-main
COPY data1.xlsx /initiativetracker-main
COPY main.py /initiativetracker-main
COPY ml.py /initiativetracker-main
COPY requirements.txt /initiativetracker-main
COPY website/ /initiativetracker-main/website/
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libopenblas-dev liblapack-dev gfortran
RUN pip install --upgrade pip
RUN pip install numpy
RUN pip install pandas
RUN pip install torch
RUN pip install Flask
RUN pip install -r requirements.txt || /bin/sh
EXPOSE 3000 
CMD ["python", "./main.py"]
