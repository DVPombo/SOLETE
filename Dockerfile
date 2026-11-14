# SOLETE platform — Dockerfile
# Author: Daniel Vázquez Pombo (daniel.vazquez.pombo@gmail.com)
# Licensed under the MIT License -- see LICENSE at the repo root.
#
# Base image matches the Python version pinned in requirements.txt: Python
# 3.13, the newest interpreter TensorFlow 2.21 currently ships wheels for.
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir --no-deps -e .

# Default command runs the example script. Swap for `CMD ["bash"]` if you'd
# rather drop into a shell and run things manually.
CMD ["python", "scripts/quickstart/RunMe.py"]
