
FROM pytorch/pytorch:latest

WORKDIR /app
COPY . /app
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --use-feature=fast-deps -r requirements.txt
RUN echo "Completed pip install"