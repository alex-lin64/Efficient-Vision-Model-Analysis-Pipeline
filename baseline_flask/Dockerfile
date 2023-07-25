FROM python:3.10.0-alpine3.13
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5000
CMD python ./index.py