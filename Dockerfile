FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

RUN apt-get update && apt-get install -y wget unzip && \
    wget -O batik_savedmodel.zip "https://github.com/YamikiiGoorden/kainara-api/releases/download/v1.2/batik_savedmodel.zip" && \
    unzip batik_savedmodel.zip && \
    rm batik_savedmodel.zip && \
    apt-get remove -y wget unzip && rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
