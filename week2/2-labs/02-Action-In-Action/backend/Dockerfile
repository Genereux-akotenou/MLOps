FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    gcc

LABEL maintainer="Généreux Akotenou <genereux.akotenou@gmail.com>"
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]