# Image de l'API : ONNX Runtime et tokenizers, sans PyTorch ni transformers (127 Mo à
# télécharger, modèle compris). Compatible avec Hugging Face Spaces, qui exige l'utilisateur 1000.
FROM python:3.11.18-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src:/app \
    PORT=8000

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Uniquement les modules utiles à la prédiction
COPY src/complaint_router/__init__.py src/complaint_router/labels.py src/complaint_router/text.py \
     src/complaint_router/predictor.py src/complaint_router/
COPY app/ app/
COPY models/ models/

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/health')"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
