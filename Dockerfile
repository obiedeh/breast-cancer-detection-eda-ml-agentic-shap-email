# Explainable medical AI demo workflow
#
# Build:
#   docker build -t medical-ai-explainability:latest .
#
# Run sample workflow:
#   docker run --rm -v $(pwd)/reports:/app/reports \
#     medical-ai-explainability:latest
#
# Run tests:
#   docker run --rm medical-ai-explainability:latest python -m pytest -q

FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt pyproject.toml README.md LICENSE ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/

RUN pip install --no-cache-dir -e ".[dev]"

VOLUME ["/app/reports"]

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import medical_ai_explainability" || exit 1

RUN useradd -m appuser && chown -R appuser /app
USER appuser

CMD ["python", "-m", "medical_ai_explainability.cli", "run-sample", "--config", "configs/default.yaml"]
