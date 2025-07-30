FROM python:3.11-slim-bullseye

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    build-essential &7 \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt \
    -f https://download.pytorch.org/whl/cpu

COPY app/ ./app/

ENV PYTORCH_ENABLE_MPS_FALLBACK=1

CMD ["python", "app/main.py"]