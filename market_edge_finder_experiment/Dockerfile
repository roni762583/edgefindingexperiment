# syntax=docker/dockerfile:1.4
# Market Edge Finder - Optimized Multi-Stage Build with Advanced Caching

# ================================
# Base Stage - Common Dependencies
# ================================
FROM python:3.11-slim as base

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG USE_CUDA=false

# Environment variables for optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Create app user and directories
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser && \
    mkdir -p /app /data /models /logs /results /cache && \
    chown -R appuser:appuser /app /data /models /logs /results /cache

# Install system dependencies with cache mounting
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libgomp1 \
        ca-certificates \
        && \
    rm -rf /var/lib/apt/lists/*

# ================================
# Dependencies Stage - Python Packages with Cache
# ================================
FROM base as deps

# Copy requirements first for optimal caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies with advanced cache mounting
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/root/.cache/torch \
    --mount=type=cache,target=/root/.cache/lightgbm \
    pip install --upgrade pip setuptools wheel && \
    pip install -r /tmp/requirements.txt

# Install PyTorch with GPU support if requested (with cache)
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$USE_CUDA" = "true" ]; then \
        pip install torch>=2.1.0+cu118 -f https://download.pytorch.org/whl/cu118/torch_stable.html; \
    fi

# ================================
# Production Stage - Minimal Runtime
# ================================
FROM base as production

# Performance optimization environment variables
ENV OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMBA_CACHE_DIR=/cache/numba \
    TORCH_HOME=/cache/torch \
    PYTHONPATH=/app:$PYTHONPATH

# Copy dependencies from deps stage
COPY --from=deps /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Set working directory
WORKDIR /app

# Copy only necessary application files (optimized layer caching)
COPY --chown=appuser:appuser . /app/

# Create cache directories with proper permissions
RUN mkdir -p /cache/numba /cache/torch /cache/pip && \
    chown -R appuser:appuser /cache

# Switch to app user for security
USER appuser

# Enhanced health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; import torch; import lightgbm; import pandas; import v20; print('✅ Health check passed'); sys.exit(0)" || exit 1

# Default production command
CMD ["python", "scripts/run_inference.py", "--daemon"]

# ================================
# Data Download Stage - Specialized for Data Operations
# ================================
FROM production as data-downloader

# Environment for data operations
ENV DATA_DOWNLOAD_MODE=true \
    LOG_LEVEL=INFO

# Volume mounts will be handled by docker-compose
VOLUME ["/data", "/logs", "/results"]

# Override entrypoint for data operations
ENTRYPOINT ["python", "scripts/run_preprocessing.py"]

# Default arguments for data download
CMD ["--download-data", "--progress", "--verify"]

# ================================
# Development Stage - Full Dev Environment
# ================================
FROM deps as development

# Install development tools with cache
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        jupyter \
        ipython \
        notebook \
        jupyterlab \
        pytest-xdist \
        black \
        mypy \
        flake8

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Switch to app user
USER appuser

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Expose Jupyter port
EXPOSE 8888

# Default command for development
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]