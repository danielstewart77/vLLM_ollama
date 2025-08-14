# ollama_proxy.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, asyncio, json, time, datetime
from typing import AsyncIterator, Dict, Any, Optional

# Upstream OpenAI-compatible server (vLLM or similar)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:7000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

app = FastAPI(title="Ollama Drop-in Proxy → vLLM (OpenAI-compatible)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True)

@app.on_event("shutdown")
async def _shutdown():
    global client
    if client:
        await client.aclose()
        client = None

# ---------- Helpers ----------

def _now_iso() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _map_ollama_options_to_openai(opts: Dict[str, Any]) -> Dict[str, Any]:
    # Map common generation knobs
    out: Dict[str, Any] = {}
    if "temperature" in opts: out["temperature"] = float(opts["temperature"])
    if "top_p" in opts:       out["top_p"] = float(opts["top_p"])
    if "max_tokens" in opts:  out["max_tokens"] = int(opts["max_tokens"])
    if "stop" in opts:        out["stop"] = opts["stop"]
    # Ollama uses "repeat_penalty"/"frequency_penalty" sometimes
    if "frequency_penalty" in opts: out["frequency_penalty"] = float(opts["frequency_penalty"])
    if "presence_penalty" in opts:  out["presence_penalty"]  = float(opts["presence_penalty"])
    if "repetition_penalty" in opts: out["frequency_penalty"] = float(opts["repetition_penalty"])  # best-effort
    return out

def _passthrough_or_json(resp: httpx.Response):
    if resp.status_code >= 400:
        return PlainTextResponse(resp.text or resp.content, status_code=resp.status_code,
                                 media_type=resp.headers.get("content-type","text/plain"))
    try:
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        return PlainTextResponse(resp.text, status_code=resp.status_code)

async def _retry_request(method: str, url: str, **kwargs) -> httpx.Response:
    assert client is not None
    max_retries, backoff = 5, 0.5
    for i in range(max_retries):
        try:
            return await client.request(method, url, **kwargs)
        except (httpx.ConnectError, httpx.ReadTimeout):
            if i == max_retries - 1:
                raise HTTPException(status_code=503, detail="Upstream not available")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 8.0)

async def _stream_openai_to_ollama_chat(
    upstream_resp: httpx.Response, model: str, started: float
) -> AsyncIterator[bytes]:
    # vLLM streams SSE: "data: {...}\n\n" with choices[0].delta.content
    total_tokens = 0
    async for raw in upstream_resp.aiter_lines():
        if not raw:
            continue
        if raw.startswith("data: "):
            data = raw[len("data: "):]
        else:
            # some servers don’t prefix; try as-is
            data = raw
        if data.strip() == "[DONE]":
            # Final Ollama-style "done" line with rough stats
            elapsed_ns = int((time.time() - started) * 1e9)
            final = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "total_duration": elapsed_ns,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": total_tokens,
                "eval_duration": elapsed_ns,
            }
            yield (json.dumps(final) + "\n").encode("utf-8")
            break
        try:
            j = json.loads(data)
        except Exception:
            # pass-through textual chunks if any
            continue
        # OpenAI chat delta path
        piece = ""
        try:
            piece = j["choices"][0].get("delta", {}).get("content", "")
            if not piece and "text" in j["choices"][0]:
                piece = j["choices"][0]["text"]
        except Exception:
            pass
        if piece:
            total_tokens += 1
            out = {
                "model": model,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": piece},
                "done": False,
            }
            yield (json.dumps(out) + "\n").encode("utf-8")

async def _stream_openai_to_ollama_generate(
    upstream_resp: httpx.Response, model: str, started: float
) -> AsyncIterator[bytes]:
    total_tokens = 0
    async for raw in upstream_resp.aiter_lines():
        if not raw:
            continue
        if raw.startswith("data: "):
            data = raw[len("data: "):]
        else:
            data = raw
        if data.strip() == "[DONE]":
            elapsed_ns = int((time.time() - started) * 1e9)
            final = {
                "model": model,
                "created_at": _now_iso(),
                "response": "",
                "done": True,
                "total_duration": elapsed_ns,
                "load_duration": 0,
                "prompt_eval_count": 0,
                "prompt_eval_duration": 0,
                "eval_count": total_tokens,
                "eval_duration": elapsed_ns,
            }
            yield (json.dumps(final) + "\n").encode("utf-8")
            break
        try:
            j = json.loads(data)
        except Exception:
            continue
        piece = ""
        try:
            # completions or chat-completions both supported
            ch = j["choices"][0]
            piece = ch.get("text") or ch.get("delta", {}).get("content", "")
        except Exception:
            pass
        if piece:
            total_tokens += 1
            out = {
                "model": model,
                "created_at": _now_iso(),
                "response": piece,
                "done": False,
            }
            yield (json.dumps(out) + "\n").encode("utf-8")

def _normalize_model_id(name: str) -> str:
    # Accept things like "model:latest" or "model:Q4_K_M"
    # vLLM typically serves the base id (left of the first colon).
    return (name or "").split(":", 1)[0]


# ---------- Ollama-compatible routes ----------

@app.get("/api/tags")
async def api_tags():
    """
    Ollama-style tags:
      { "models": [ { "name": "model:tag", "model": "model", ... }, ... ] }
    We map from /v1/models ids and synthesize minimal metadata.
    """
    r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
    try:
        data = r.json()
    except Exception:
        return PlainTextResponse(r.text, status_code=r.status_code)

    models = []
    for m in data.get("data", []):
        name = m.get("id", "")
        if not name:
            continue
        base = name.split(":", 1)[0]  # best-effort base name
        models.append({
            "name": name,
            "model": base,
            "modified_at": _now_iso(),
            "size": 0,
            "digest": "",
            "details": {"parent_model": "", "format": "pytorch", "families": []},
        })
    return JSONResponse({"models": models}, status_code=r.status_code)

# Some UIs call /api/models; mirror /api/tags for compatibility.
@app.get("/api/models")
async def api_models():
    return await api_tags()


@app.get("/api/version")
async def api_version():
    # Mimic Ollama's version format
    return {"version": "0.1.0-proxy"}

@app.get("/api/ps")
async def api_ps():
    # Minimal process list; Ollama returns running models.
    # We'll map to available models for now.
    # If you want true "running" semantics, we can track active sessions.
    r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
    try:
        data = r.json()
    except Exception:
        # Pass upstream error through
        return PlainTextResponse(r.text, status_code=r.status_code)
    procs = [{"model": m.get("id", ""), "size": 0, "digest": ""} for m in data.get("data", [])]
    return {"models": procs}

@app.post("/api/show")
async def api_show(request: Request):
    # Ollama usually accepts {"name":"model"} and returns metadata.
    body = await request.json()
    name = body.get("name") or body.get("model") or ""
    if not name:
        return JSONResponse({"error": "missing model name"}, status_code=400)
    # Best-effort metadata
    return {
        "model": name,
        "modified_at": _now_iso(),
        "size": 0,
        "digest": "",
        "details": {"parent_model": "", "format": "pytorch", "families": []},
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    """
    Ollama-style request:
    {
      "model": "granite-8b",
      "messages": [{"role":"system","content":"..."}, {"role":"user","content":"..."}],
      "stream": true,
      "options": { "temperature": 0.2, "top_p": 0.95, ... }
    }
    Streamed response: NDJSON lines with {"message":{"role":"assistant","content":"..."}, "done": false}
    Final line includes {"done": true, stats...}
    """
    body = await request.json()
    model_in = body.get("model", "")
    model = _normalize_model_id(model_in)
    messages = body.get("messages", [])
    stream = bool(body.get("stream", True))
    options = body.get("options", {}) or {}

    openai_payload = {
        "model": model,
        "messages": messages,
        **_map_ollama_options_to_openai(options),
    }

    if stream:
        openai_payload["stream"] = True
        upstream = await _retry_request(
            "POST", f"{VLLM_BASE_URL}/v1/chat/completions",
            json=openai_payload, headers={"accept": "text/event-stream"}
        )
        started = time.time()
        return StreamingResponse(
            _stream_openai_to_ollama_chat(upstream, model, started),
            media_type="application/x-ndjson"
        )
    else:
        upstream = await _retry_request(
            "POST", f"{VLLM_BASE_URL}/v1/chat/completions",
            json=openai_payload
        )
        try:
            data = upstream.json()
        except Exception:
            return PlainTextResponse(upstream.text, status_code=upstream.status_code)

        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        # Non-stream one-shot Ollama-style response
        resp = {
            "model": model,
            "created_at": _now_iso(),
            "message": {"role": "assistant", "content": content},
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": len(content.split()),
            "eval_duration": 0,
        }
        return JSONResponse(resp, status_code=upstream.status_code)

@app.post("/api/generate")
async def api_generate(request: Request):
    """
    Ollama-style request:
    {
      "model": "granite-8b",
      "prompt": "Write a haiku",
      "stream": true,
      "options": { ... }
    }
    """
    body = await request.json()
    model_in = body.get("model", "")
    model = _normalize_model_id(model_in)
    prompt = body.get("prompt", "")
    stream = bool(body.get("stream", True))
    options = body.get("options", {}) or {}

    # Prefer /v1/completions for plain prompts. If missing upstream, we can fallback to chat.
    openai_payload = {
        "model": model,
        "prompt": prompt,
        **_map_ollama_options_to_openai(options),
    }

    if stream:
        openai_payload["stream"] = True
        upstream = await _retry_request(
            "POST", f"{VLLM_BASE_URL}/v1/completions",
            json=openai_payload, headers={"accept": "text/event-stream"}
        )
        started = time.time()
        return StreamingResponse(
            _stream_openai_to_ollama_generate(upstream, model, started),
            media_type="application/x-ndjson"
        )
    else:
        upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/completions", json=openai_payload)
        try:
            data = upstream.json()
        except Exception:
            return PlainTextResponse(upstream.text, status_code=upstream.status_code)

        text = data.get("choices", [{}])[0].get("text", "")
        resp = {
            "model": model,
            "created_at": _now_iso(),
            "response": text,
            "done": True,
            "total_duration": 0,
            "load_duration": 0,
            "prompt_eval_count": 0,
            "prompt_eval_duration": 0,
            "eval_count": len(text.split()),
            "eval_duration": 0,
        }
        return JSONResponse(resp, status_code=upstream.status_code)

@app.post("/api/embeddings")
async def api_embeddings(request: Request):
    """
    Ollama-style request:
    { "model": "granite-embed", "input": "text" }
    Ollama also accepts { "prompt": "..." }. We'll accept either.
    """
    body = await request.json()
    model_in = body.get("model", "")
    model = _normalize_model_id(model_in)
    inp = body.get("input") or body.get("prompt") or ""
    if isinstance(inp, list):
        input_list = inp
    else:
        input_list = [inp]

    upstream = await _retry_request(
        "POST", f"{VLLM_BASE_URL}/v1/embeddings",
        json={"model": model, "input": input_list}
    )
    try:
        data = upstream.json()
    except Exception:
        return PlainTextResponse(upstream.text, status_code=upstream.status_code)

    # OpenAI format: {data:[{embedding:[...]}]}
    embeddings = [d.get("embedding", []) for d in data.get("data", [])]
    # Ollama returns a single vector for single input
    if len(embeddings) == 1:
        return JSONResponse({"model": model, "embedding": embeddings[0]}, status_code=upstream.status_code)
    else:
        return JSONResponse({"model": model, "embeddings": embeddings}, status_code=upstream.status_code)

# Optional health
@app.get("/healthz")
async def healthz():
    try:
        r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
        return JSONResponse({"ok": True, "upstream": r.status_code})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
