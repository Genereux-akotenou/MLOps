FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc

WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=200 -r requirements.txt
COPY . .

EXPOSE 8001
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8001"]