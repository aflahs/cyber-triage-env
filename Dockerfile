# Cyber Threat Intelligence Triage Environment — OpenEnv
# Hugging Face Spaces compatible Dockerfile

FROM python:3.11-slim

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (cache layer)
COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user . /app

# Install the package
RUN pip install --no-cache-dir -e .

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# Start the server
CMD ["uvicorn", "cyber_triage_env.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
