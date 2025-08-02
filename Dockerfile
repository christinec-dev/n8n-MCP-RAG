FROM python:3.11-slim

# System deps (dos2unix helps if files came from Windows)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git ca-certificates dos2unix \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# App code
COPY . ./

# Fix CRLF line endings on Windows scripts (no-op on Linux)
RUN dos2unix entrypoint.sh || true && chmod +x entrypoint.sh

EXPOSE 8000
ENV DB_DIR=/app/chroma_db
ENV RUN_INDEX_ON_START=true

ENTRYPOINT ["./entrypoint.sh"]
