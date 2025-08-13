# vLLM OpenAI & Ollama Compatible API Server

A high-performance inference server that provides both OpenAI-compatible and Ollama-compatible APIs for the IBM Granite 3.3 8B Instruct model, powered by vLLM for optimized GPU acceleration.

## 🚀 Features

- **Dual API Compatibility**: Supports both OpenAI and Ollama API formats
- **High Performance**: Optimized for NVIDIA RTX A5000 with GPU memory utilization
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **Streaming Support**: Real-time streaming responses for chat and completions
- **Model Caching**: Persistent Hugging Face model cache
- **Retry Logic**: Built-in connection retry with exponential backoff
- **Health Monitoring**: Health check endpoints for monitoring

## 📋 Prerequisites

- NVIDIA GPU (RTX A5000 recommended, 24GB VRAM)
- NVIDIA Docker runtime
- Python 3.8+ (for local installation)
- CUDA 12.1+

## 🛠️ Installation & Setup

### Option 1: Docker (Recommended)

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd vLLM
   ```

2. **Run with Docker Compose:**
   ```bash
   # For Ollama-compatible API
   docker-compose -f docker-compose.ollama.yml up -d
   
   # For OpenAI-compatible API
   docker-compose up -d
   ```

### Option 2: Local Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the vLLM server:**
   ```bash
   # Make the script executable
   chmod +x run_vllm_ollama.sh
   
   # Start the server
   ./run_vllm_ollama.sh
   ```

## 🔧 Configuration

### GPU Optimization Settings

The server is pre-configured for optimal performance on RTX A5000:

- **Model**: IBM Granite 3.3 8B Instruct
- **GPU Memory Utilization**: 90%
- **Max Model Length**: 16,384 tokens
- **Data Type**: bfloat16 (optimized for Ampere architecture)
- **Batch Processing**: Up to 4 sequences with 16,384 batched tokens

### Environment Variables

- `VLLM_BASE_URL`: vLLM server URL (default: `http://localhost:7000`)
- `REQUEST_TIMEOUT`: Request timeout in seconds (default: `600`)
- `CUDA_DEVICE_ORDER`: GPU device ordering (set to `PCI_BUS_ID`)

## 📡 API Endpoints

### OpenAI Compatible API (Port 3000)

- `GET /v1/models` - List available models
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions
- `POST /v1/embeddings` - Text embeddings

### Ollama Compatible API (Port 3000)

- `POST /api/chat` - Ollama-style chat
- `POST /api/generate` - Ollama-style text generation
- `POST /api/embeddings` - Ollama-style embeddings
- `GET /api/tags` - List available models
- `GET /healthz` - Health check

### vLLM Server (Port 7000)

Direct access to the vLLM OpenAI-compatible server.

## 💡 Usage Examples

### OpenAI API Style

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:3000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="ibm-granite/granite-3.3-8b-instruct",
    messages=[
        {"role": "user", "content": "Hello, how are you?"}
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")
```

### Ollama API Style

```bash
# Chat completion
curl -X POST http://localhost:3000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-8b",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": true
  }'

# Simple generation
curl -X POST http://localhost:3000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "granite-8b",
    "prompt": "Write a haiku about AI",
    "stream": false
  }'
```

## 🐳 Docker Configuration

### Ollama-Compatible Container

```yaml
services:
  openai-proxy:
    build:
      context: .
      dockerfile: Dockerfile.ollama
    ports:
      - "3000:3000"  # FastAPI proxy
      - "7000:7000"  # vLLM server
    volumes:
      - ./hf_cache:/root/.cache/huggingface
    runtime: nvidia
```

### Performance Tuning

The configuration includes several optimization strategies:

1. **Speed Optimization**: Prioritizes inference speed with large batch sizes
2. **Memory Optimization**: Maximizes GPU memory utilization (90%)
3. **Precision Balance**: Uses bfloat16 for optimal speed/accuracy trade-off
4. **Tool Use Ready**: Maintains sufficient precision for function calling

## 🔍 Monitoring & Health Checks

- **Health Endpoint**: `GET /healthz` returns server status
- **Model Listing**: `GET /v1/models` or `GET /api/tags`
- **Logs**: Check container logs for performance metrics

## 🚨 Troubleshooting

### Common Issues

1. **GPU Memory Errors**: Reduce `max-num-seqs` or `gpu-memory-utilization`
2. **Connection Timeouts**: Increase `REQUEST_TIMEOUT` environment variable
3. **Model Loading**: Ensure sufficient disk space for model cache

### Performance Tips

- For maximum throughput: Remove `--enforce-eager` flag
- For tool calling: Use `float16` if bfloat16 produces formatting errors
- For single requests: Keep `max-num-seqs` low (4-8)
- For high concurrency: Increase `max-num-seqs` (64-128)

## 📦 Model Information

- **Model**: IBM Granite 3.3 8B Instruct
- **Size**: ~16GB (bfloat16)
- **Context Length**: 16,384 tokens
- **Architecture**: Optimized for instruction following and tool use

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🔗 Related Projects

- [vLLM](https://github.com/vllm-project/vllm) - High-throughput LLM inference engine
- [Ollama](https://ollama.ai/) - Local LLM runner
- [IBM Granite](https://huggingface.co/ibm-granite) - Enterprise-grade language models

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Review vLLM documentation for advanced configuration
- Open an issue on GitHub for bugs or feature requests
