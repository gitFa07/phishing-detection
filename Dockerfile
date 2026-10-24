FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt

COPY src/ src/
COPY models/ models/
COPY frontend/ frontend/

RUN useradd --create-home appuser
USER appuser

# Render (and most hosts) pass the port in $PORT.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn src.app:app --host 0.0.0.0 --port ${PORT}"]
