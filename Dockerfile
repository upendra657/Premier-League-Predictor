# syntax=docker/dockerfile:1
#
# Multi-stage build for the Premier League Decision Engine inference service.
# The builder compiles wheels; the runtime carries only the installed packages,
# the source tree and the model artifacts. Training dependencies and the raw
# data never enter the shipped image.

# --------------------------------------------------------------------------- #
# Stage 1 -- build dependency wheels
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements-serve.txt

# --------------------------------------------------------------------------- #
# Stage 2 -- runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user: a container that never needs to write to its
# own filesystem should not be able to.
RUN useradd --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artifacts/ ./artifacts/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
p=os.environ.get('PORT','8000'); \
sys.exit(0 if urllib.request.urlopen(f'http://localhost:{p}/health').status==200 else 1)"

# Shell form so ${PORT} expands: Render, Railway and Fly inject the port to
# bind, and a hardcoded 8000 means the container never receives traffic. exec
# keeps uvicorn as PID 1 so SIGTERM reaches it and shutdown stays graceful.
#
# One worker, not two. Each worker holds its own copy of pandas, scikit-learn,
# xgboost and the booster -- measured at 204MB RSS. Two would need ~408MB
# against the 512MB a free tier allows, and OOM under any real load.
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WEB_CONCURRENCY:-1}"]
