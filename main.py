from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import httpx
import os
import asyncio

# Local vLLM OpenAI-compatible server
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:7000")

app = FastAPI()

client = httpx.AsyncClient(timeout=60.0)

async def make_request_with_retry(method: str, url: str, **kwargs) -> httpx.Response:
    """Make HTTP request with retry logic for connection errors"""
    max_retries = 5
    for attempt in range(max_retries):
        try:
            if method.upper() == "GET":
                response = await client.get(url, **kwargs)
            else:
                response = await client.post(url, **kwargs)
            return response
        except httpx.ConnectError:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                continue
            # If all retries failed, raise an exception
            raise HTTPException(status_code=503, detail="vLLM API server not available")
    # This should never be reached, but adding for type safety
    raise HTTPException(status_code=503, detail="vLLM API server not available")

@app.get("/v1/models")
async def models():
    response = await make_request_with_retry("GET", f"{VLLM_BASE_URL}/v1/models")
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    response = await make_request_with_retry("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/v1/completions")
async def completions(request: Request):
    body = await request.json()
    response = await make_request_with_retry("POST", f"{VLLM_BASE_URL}/v1/completions", json=body)
    return JSONResponse(content=response.json(), status_code=response.status_code)

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    body = await request.json()
    response = await make_request_with_retry("POST", f"{VLLM_BASE_URL}/v1/embeddings", json=body)
    return JSONResponse(content=response.json(), status_code=response.status_code)
