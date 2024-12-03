FROM python:3.7-slim-buster

WORKDIR /run/media/thunderrr/Code/WorkSpace/Database/ML_Deployment

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "dataPipeline.py"]