# Base image
FROM python:3.7-slim-buster

WORKDIR /run/media/thunderrr/Code/WorkSpace/Database/ML_Deployment

# installing system dependencies
RUN apt-get update && apt-get install -y libpq-dev

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
# Copying the application code
COPY . .
CMD ["python", "data_Pipeline.py"]