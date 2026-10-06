# Serving image: the API with the model copied in at build time, so the
# running container needs no credentials.
#
#   python -m src.export_champion --remote    # writes serving_model/
#   docker build -t movie-rec .
#   docker run -p 8000:8000 movie-rec
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/serving_model

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY serving_model/ serving_model/

# One linear-algebra thread per request. Scoring a user is a small matrix product; with the
# default (one thread per core) the threads cost about ten times more than the product itself.
ENV OPENBLAS_NUM_THREADS=1     OMP_NUM_THREADS=1

RUN useradd --create-home --uid 1000 app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
