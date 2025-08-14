#!/bin/bash

# Set environment variable for CUDA device ordering
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Speed-optimized settings for RTX A5000 (24GB VRAM) - prioritizing inference speed over concurrency
python3 -m vllm.entrypoints.openai.api_server \
  --model ibm-granite/granite-3.3-8b-instruct \
  --port 7000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.50 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 16384 \
  --disable-log-stats \
  --enforce-eager \
  --dtype bfloat16 \
  --block-size 32 \
  --swap-space 4 \
  --tensor-parallel-size 2 \
  &

# For dual GPU setup (A5000 + A5000), uncomment:
  

sleep 10

uvicorn ollama_api:app --host 0.0.0.0 --port 3000

# Speed Optimization Analysis 🚀
# Your goal is to maximize inference speed. This can mean two different things:

# Latency: The time it takes to complete a single request (lowest time-to-first-token).

# Throughput: The total number of requests processed per second.

# Your current settings are a bit of a mix between these two goals. Here are the key points and a recommended revision.

# What You're Doing Right
# dtype bfloat16: Excellent choice. The RTX A5000 (Ampere architecture) is highly optimized for bfloat16 operations, offering a great balance of speed and precision.

# gpu-memory-utilization 0.98: Good. This allocates as much VRAM as possible to the KV cache, allowing for larger batches and longer sequences, which boosts throughput.

# max-num-batched-tokens 16384: Perfect. Setting this to a high value allows vLLM to create large, efficient batches, which is key for GPU utilization and overall throughput.

# Key Changes for Better Speed
# Remove --enforce-eager. This is the most important change. While "eager mode" can sometimes be faster for single, small requests by skipping a compilation step, it prevents vLLM from using CUDA graphs. For sustained workloads, CUDA graphs significantly reduce CPU overhead and can lead to a 1.2x to 2x speedup after a brief warm-up period. Since your goal is speed, you want CUDA graphs enabled (which is the default behavior when you remove this flag).

# Increase --max-num-seqs. Your current setting of 4 severely limits concurrency and works against your goal of high throughput. It tells vLLM to only handle 4 sequences at a time. Unless you are only ever sending one request at a time and care purely about single-request latency, this is a bottleneck. Increase it to allow vLLM to build larger batches. A value like 128 is a much better starting point for a server.

# Recommended Speed-Optimized Configuration
# This configuration removes the bottlenecks and fully leverages vLLM's batching and CUDA graph capabilities for maximum throughput.

# Bash

# #!/bin/bash

# # Set environment variable for CUDA device ordering
# export CUDA_DEVICE_ORDER=PCI_BUS_ID

# # Speed-optimized settings for RTX A5000 (24GB VRAM)
# # This configuration prioritizes high throughput.
# python3 -m vllm.entrypoints.openai.api_server \
#   --model ibm-granite/granite-3.3-8b-instruct \
#   --port 7000 \
#   --max-model-len 16384 \
#   --gpu-memory-utilization 0.98 \
#   --max-num-seqs 128 \
#   --max-num-batched-tokens 16384 \
#   --dtype bfloat16 \
#   --block-size 32 \
#   --disable-log-stats \
#   --swap-space 4 \
#   &

# sleep 10

# uvicorn main:app --host 0.0.0.0 --port 3000
# Tool Use Optimization Analysis 🛠️
# When you optimize for tool use (or function calling), your primary concern shifts slightly from raw speed to generation fidelity. The model must generate text that adheres to a very strict format (like JSON schema). If the output is even slightly off, the tool call will fail.

# The main parameter that influences this is model precision (dtype).

# Your Current Setting (bfloat16): This is usually sufficient. Most modern models are trained with bfloat16 and perform well with tool use at this precision. You should start here and only change it if you observe failures.

# When to Change: If you find the model is frequently generating malformed JSON or failing to follow formatting instructions for tool calls, it might be due to a loss of precision.

# How the Configuration Would Change
# If bfloat16 proves insufficient, you would need to increase the precision. This will come at the cost of speed and memory.

# First Alternative (float16): Change --dtype bfloat16 to --dtype float16. This offers more precision than bfloat16 but is slightly slower on your GPU.

# Last Resort (float32): Change to --dtype float32. This provides full precision but will be significantly slower and will use twice the VRAM, drastically reducing the number of concurrent sequences you can handle. You would likely need to lower --max-num-seqs and --max-num-batched-tokens.

# All other parameters (like --max-num-seqs, --max-num-batched-tokens, etc.) should remain the same as the speed-optimized configuration, as they relate to server workload, not the quality of a single generation.

# In summary, for tool use:

# Start with the speed-optimized config (using bfloat16).

# Test your tool-calling prompts rigorously.

# Only increase dtype precision to float16 or float32 if you encounter persistent formatting errors.
