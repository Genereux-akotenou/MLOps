FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]