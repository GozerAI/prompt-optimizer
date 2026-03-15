FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/* && \
    useradd -r -s /bin/false appuser

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[api]"

USER appuser

EXPOSE 8013

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8013/health || exit 1

CMD ["uvicorn", "prompt_optimizer.api:app", "--host", "0.0.0.0", "--port", "8013"]
