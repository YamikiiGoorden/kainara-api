FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN apt-get update && apt-get install -y wget && \
    wget -O batik_resnet50.h5 "https://github.com/YamikiiGoorden/kainara-api/releases/download/v1.0/batik_resnet50.h5" && \
    apt-get remove -y wget && rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
