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

# The full SOLETE dataset itself is not included in the image (see README) --
# mount it at runtime, e.g. `docker run --rm -v $(pwd)/data:/app/data solete`.
# The small SOLETE_short.h5 sample IS baked into the image (see .dockerignore).

# Default command runs the example script. Swap for `CMD ["bash"]` if you'd
# rather drop into a shell and run things manually.
CMD ["python", "RunMe.py"]
