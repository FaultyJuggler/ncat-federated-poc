FROM nvcr.io/nvidia/rapids:23.12-cuda11.8-runtime-ubuntu22.04-py3.10

# Rest of your Dockerfile

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create symlinks for Python
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install Python packages
COPY central_server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install psutil

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

EXPOSE 8080

CMD ["python", "server.py"]