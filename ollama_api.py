# ollama_proxy.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx, os, asyncio, json, time, datetime
from typing import AsyncIterator, Dict, Any, Optional

# Upstream OpenAI-compatible server (vLLM or similar)
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:7000")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

app = FastAPI(title="Ollama + OpenAI Drop-in Proxy → vLLM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

client: Optional[httpx.AsyncClient] = None
_last_calls = []  # tiny in-memory ring for quick debugging

@app.middleware("http")
async def _log_every_request(request: Request, call_next):
    try:
        body = None
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.body()
        _last_calls.append({
            "ts": time.time(),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query),
            "body": (body.decode("utf-8", errors="ignore") if body else None)[:512],
        })
        # keep last 30
        if len(_last_calls) > 30:
            del _last_calls[:-30]
        # re-inject body for downstream handlers
        async def bodygen():
            if body is not None:
                yield body
        request._receive = lambda: asyncio.get_event_loop().create_task(asyncio.sleep(0))  # type: ignore
        response = await call_next(request)
        return response
    except Exception:
        return await call_next(request)

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
    return datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat().replace("+00:00", "Z")

def _normalize_model_id(name: str) -> str:
    # Accept "model:latest" or "model:anytag" → "model"
    return (name or "").split(":", 1)[0]

def _map_ollama_options_to_openai(opts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "temperature" in opts: out["temperature"] = float(opts["temperature"])
    if "top_p" in opts:       out["top_p"] = float(opts["top_p"])
    if "max_tokens" in opts:  out["max_tokens"] = int(opts["max_tokens"])
    if "stop" in opts:        out["stop"] = opts["stop"]
    if "frequency_penalty" in opts: out["frequency_penalty"] = float(opts["frequency_penalty"])
    if "presence_penalty" in opts:  out["presence_penalty"]  = float(opts["presence_penalty"])
    if "repetition_penalty" in opts: out["frequency_penalty"] = float(opts["repetition_penalty"])  # best-effort
    return out

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

def _passthrough_or_json(resp: httpx.Response):
    if resp.status_code >= 400:
        return PlainTextResponse(
            resp.text or resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type", "text/plain"),
        )
    try:
        return JSONResponse(resp.json(), status_code=resp.status_code)
    except Exception:
        return PlainTextResponse(resp.text, status_code=resp.status_code)

# ----- Stream translators (OpenAI SSE → Ollama NDJSON) -----

async def _stream_openai_to_ollama_chat(
    upstream_resp: httpx.Response, model: str, started: float
) -> AsyncIterator[bytes]:
    total_tokens = 0
    async for raw in upstream_resp.aiter_lines():
        if not raw:
            continue
        data = raw[6:] if raw.startswith("data: ") else raw
        if data.strip() == "[DONE]":
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
            continue
        piece = j.get("choices", [{}])[0].get("delta", {}).get("content", "") or \
                j.get("choices", [{}])[0].get("text", "")
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
        data = raw[6:] if raw.startswith("data: ") else raw
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
        piece = j.get("choices", [{}])[0].get("text") or \
                j.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if piece:
            total_tokens += 1
            out = {
                "model": model,
                "created_at": _now_iso(),
                "response": piece,
                "done": False,
            }
            yield (json.dumps(out) + "\n").encode("utf-8")

# ---------- Ollama-compatible routes ----------

@app.get("/api/version")
async def api_version():
    return {"version": "0.1.0-proxy"}

@app.get("/api/ps")
async def api_ps():
    r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
    if r.status_code >= 400:
        return _passthrough_or_json(r)
    data = r.json()
    procs = [{"model": m.get("id", ""), "size": 0, "digest": ""} for m in data.get("data", [])]
    return {"models": procs}

@app.get("/api/tags")
async def api_tags():
    r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
    if r.status_code >= 400:
        return _passthrough_or_json(r)

    models = []
    for m in r.json().get("data", []):
        mid = m.get("id", "")
        if not mid:
            continue
        base = _normalize_model_id(mid)
        pretty = f"{base}:latest" if ":" not in mid else mid
        models.append({
            "name": pretty,
            "model": base,
            "modified_at": _now_iso(),
            "size": 0,
            "digest": "",
            "details": {"parent_model": "", "format": "pytorch", "families": []},
        })
    return {"models": models}

@app.get("/api/models")
async def api_models():
    return await api_tags()

@app.post("/api/show")
async def api_show(request: Request):
    body = await request.json()
    name = body.get("name") or body.get("model") or ""
    if not name:
        return JSONResponse({"error": "missing model name"}, status_code=400)
    return {
        "model": name,
        "modified_at": _now_iso(),
        "size": 0,
        "digest": "",
        "details": {"parent_model": "", "format": "pytorch", "families": []},
    }

@app.post("/api/chat")
async def api_chat(request: Request):
    body = await request.json()
    model = _normalize_model_id(body.get("model", ""))
    messages = body.get("messages", [])
    stream = bool(body.get("stream", True))
    options = body.get("options", {}) or {}
    payload = {"model": model, "messages": messages, **_map_ollama_options_to_openai(options)}

    if stream:
        payload["stream"] = True
        upstream = await _retry_request(
            "POST", f"{VLLM_BASE_URL}/v1/chat/completions",
            json=payload, headers={"accept": "text/event-stream"}
        )
        started = time.time()
        return StreamingResponse(_stream_openai_to_ollama_chat(upstream, model, started),
                                 media_type="application/x-ndjson")
    else:
        upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=payload)
        return _passthrough_or_json(upstream)  # let client parse OpenAI object or error

@app.post("/api/generate")
async def api_generate(request: Request):
    body = await request.json()
    model = _normalize_model_id(body.get("model", ""))
    prompt = body.get("prompt", "")
    stream = bool(body.get("stream", True))
    options = body.get("options", {}) or {}
    payload = {"model": model, "prompt": prompt, **_map_ollama_options_to_openai(options)}

    if stream:
        payload["stream"] = True
        upstream = await _retry_request(
            "POST", f"{VLLM_BASE_URL}/v1/completions",
            json=payload, headers={"accept": "text/event-stream"}
        )
        started = time.time()
        return StreamingResponse(_stream_openai_to_ollama_generate(upstream, model, started),
                                 media_type="application/x-ndjson")
    else:
        upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/completions", json=payload)
        return _passthrough_or_json(upstream)

@app.post("/api/embeddings")
async def api_embeddings(request: Request):
    body = await request.json()
    model = _normalize_model_id(body.get("model", ""))
    inp = body.get("input") or body.get("prompt") or ""
    input_list = inp if isinstance(inp, list) else [inp]
    upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/embeddings",
                                    json={"model": model, "input": input_list})
    return _passthrough_or_json(upstream)

# ---------- EXTRA: Open WebUI sometimes calls /api/chat/completions (OpenAI shape under /api)
@app.post("/api/chat/completions")
async def api_chat_completions_alias(request: Request):
    body = await request.json()
    if "model" in body:
        body["model"] = _normalize_model_id(body["model"])
    upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
    return _passthrough_or_json(upstream)

@app.post("/api/completions")
async def api_completions_alias(request: Request):
    body = await request.json()
    if "model" in body:
        body["model"] = _normalize_model_id(body["model"])
    upstream = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/completions", json=body)
    return _passthrough_or_json(upstream)

# ---------- Health & debug ----------

@app.get("/healthz")
async def healthz():
    try:
        r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
        return JSONResponse({"ok": True, "upstream": r.status_code})
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

@app.get("/_debug")
async def _debug():
    return {"last_calls": _last_calls[-10:]}

# ---------- OpenAI-compatible routes (for clients that use /v1/*) ----------

@app.get("/v1/models")
async def v1_models():
    r = await _retry_request("GET", f"{VLLM_BASE_URL}/v1/models")
    return _passthrough_or_json(r)

@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    body = await request.json()
    if "model" in body:
        body["model"] = _normalize_model_id(body["model"])
    r = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/chat/completions", json=body)
    return _passthrough_or_json(r)

@app.post("/v1/completions")
async def v1_completions(request: Request):
    body = await request.json()
    if "model" in body:
        body["model"] = _normalize_model_id(body["model"])
    r = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/completions", json=body)
    return _passthrough_or_json(r)

@app.post("/v1/embeddings")
async def v1_embeddings(request: Request):
    body = await request.json()
    if "model" in body:
        body["model"] = _normalize_model_id(body["model"])
    r = await _retry_request("POST", f"{VLLM_BASE_URL}/v1/embeddings", json=body)
    return _passthrough_or_json(r)
