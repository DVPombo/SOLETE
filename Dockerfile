# SOLETE platform — Dockerfile
# Author: Daniel Vázquez Pombo (daniel.vazquez.pombo@gmail.com)
# Licensed under the MIT License -- see LICENSE at the repo root.
#
# UNTESTED: `docker build` was not run against this Dockerfile (Docker is not
# available in the environment this was authored in). Please verify with
# `docker build -t solete .` before relying on it.
#
# Base image matches the Python version pinned in requirements.txt: Python
# 3.13, the newest interpreter TensorFlow 2.21 currently ships wheels for.
FROM python:3.13-slim

WORKDIR /app

# Build tools kept as a safety margin in case any transitive dependency still
# needs to compile from source on your target platform/arch; all direct
# dependencies installed as prebuilt wheels when this was last checked.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# The full SOLETE dataset itself is not included in the image (see data/README.md) --
# mount your data folder (with hdf5/ and parquet/ inside) at runtime:
#     docker run --rm -v $(pwd)/data:/app/data solete
# solete/paths.py finds it at /app/data. The small examples/SOLETE_short.h5 sample IS
# baked into the image (see .dockerignore).

# Default command runs the example script. Swap for `CMD ["bash"]` if you'd
# rather drop into a shell and run things manually.
CMD ["python", "scripts/quickstart/RunMe.py"]
