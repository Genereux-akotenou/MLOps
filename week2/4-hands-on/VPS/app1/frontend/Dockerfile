FROM python:3.10-slim

LABEL maintainer="Généreux Akotenou <genereux.akotenou@gmail.com>"

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

CMD ["streamlit", "run", "frontend.py", "--server.port", "8501", "--server.address", "0.0.0.0"]