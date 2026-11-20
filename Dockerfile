FROM python:3.11-slim

# TF_CPP_MIN_LOG_LEVEL=2 hides TensorFlow's info/warning startup noise.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# Install deps first so this layer is cached when only app code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run as a non-root user: a compromised process cannot modify the image or system files.
RUN useradd --create-home --uid 1000 appuser
COPY app ./app
COPY artifacts ./artifacts
# Measured results shown on the Performance and Model quality pages (/metrics/evaluation).
COPY benchmarks/results ./benchmarks/results
COPY evaluation/results ./evaluation/results
# SQLite prediction log; the only folder the app writes to.
RUN mkdir -p /app/data && chown appuser /app/data
USER appuser

EXPOSE 8000

# Uses Python's stdlib (slim image has no curl). Start period covers TF import + model load.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os, urllib.request as u; u.urlopen('http://localhost:%s/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1

# sh -c expands $PORT (Render injects its own); exec makes uvicorn PID 1 so it receives SIGTERM.
# Single worker: each worker would hold its own copy of the model in memory.
# X-Forwarded-For is deliberately not trusted (clients can fake it); the rate limiter reads the
# visitor IP from CLIENT_IP_HEADER instead, set per platform (see render.yaml).
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
