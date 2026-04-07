# Cyber Threat Intelligence Triage Environment — OpenEnv
# Hugging Face Spaces compatible Dockerfile

FROM python:3.11-slim

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY env/           ./env/
COPY data/          ./data/
COPY server.py      .
COPY inference.py   .
COPY openenv.yaml   .

# Set ownership
RUN chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Environment variable defaults (override at runtime)
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:7860/health').raise_for_status()"

CMD ["python", "server.py"]
