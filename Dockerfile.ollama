FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install Python and basic packages
RUN apt-get update && apt-get install -y \
    python3-pip python3-dev git curl && \
    ln -sf python3 /usr/bin/python && \
    pip install --upgrade pip

# Install app dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY ollama_api.py /app/ollama_api.py
COPY run_vllm_ollama.sh /app/run_vllm_ollama.sh
WORKDIR /app

# Set permissions
RUN chmod +x /app/run_vllm_ollama.sh

# Expose FastAPI and vLLM ports
EXPOSE 3000 7000

CMD ["bash", "/app/run_vllm_ollama.sh"]
