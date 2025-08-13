# ---------------------------
# Prompt refinement endpoint (must be after app = FastAPI(...))
# ---------------------------
from fastapi import Body

# mcp_server.py
import re
import functools
import hashlib
import json
import langwatch
import os, subprocess, sys
from typing import List, Tuple, Dict
from fastapi import FastAPI, Header, Query, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langwatch.types import RAGChunk

ALLOW_UI_KEYS = os.getenv("ALLOW_UI_KEYS", "false").lower() in ("1","true","yes")

# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="n8n RAG Server")

# ---------------------------
# Prompt refinement endpoint (must be after app = FastAPI(...))
# ---------------------------
from fastapi import Body

@app.post("/refine_prompt")
def refine_prompt(prompt: str = Body(..., embed=True)):
    """Refine/rewrite a prompt using the LLM-based rewrite_prompt function."""
    try:
        refined = rewrite_prompt(prompt)
        return {"refined": refined}
    except Exception as e:
        return {"error": str(e)}
MIN_CONTEXT_HITS = int(os.getenv("MIN_CONTEXT_HITS", "2"))
REQUIRE_CONTEXT  = os.getenv("REQUIRE_CONTEXT","false").lower() in ("1","true","yes")
QUERY_EXPANSION  = os.getenv("QUERY_EXPANSION","true").lower() in ("1","true","yes")
RETRIEVE_K       = int(os.getenv("RETRIEVE_K", "10"))

if os.getenv("LANGWATCH_API_KEY"):
    # optional: LANGWATCH_ENDPOINT if you self-host, else defaults to cloud
    langwatch.setup()  # reads LANGWATCH_API_KEY / LANGWATCH_ENDPOINT from env

def _truncate(s: str, n: int = 4000) -> str:
    return s if s is None or len(s) <= n else s[:n] + "…"

def expand_queries(q: str) -> list[str]:
    if not QUERY_EXPANSION:
        return []
    # cheap heuristic expansions – avoids another model call
    terms = []
    low = q.lower()
    if "webhook" in low: terms += ["HTTP Webhook", "Respond to Webhook", "HTTP Request"]
    if "airtable" in low: terms += ["Airtable Base", "Airtable API", "Upsert to Airtable"]
    if "slack" in low: terms += ["Slack Post Message", "Slack Bot"]
    if "gmail" in low or "email" in low: terms += ["Gmail", "IMAP Email Read", "SMTP Send"]
    if "cron" in low or "daily" in low or "schedule" in low: terms += ["Schedule Trigger", "Cron"]
    return [q] + list(dict.fromkeys(terms))  # dedupe


# ---------------------------
# Chroma client & collections
# ---------------------------
import chromadb
DB_DIR = "./chroma_db"

# Handle Chroma 0.5+ (PersistentClient) and 0.4.x
try:
    from chromadb import PersistentClient  # type: ignore
    client = PersistentClient(path=DB_DIR)
except Exception:
    from chromadb.config import Settings  # type: ignore
    client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=DB_DIR)
    )

def _get_collection(name: str):
    try:
        return client.get_collection(name)
    except Exception:
        # If index not built yet, create an empty one so server still runs
        return client.get_or_create_collection(name)

WF   = _get_collection("n8n-workflows")
DOCS = _get_collection("n8n-docs")

# ---------------------------
# Token utilities
# ---------------------------
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def tcount(s: str) -> int:
    return len(enc.encode(s))

# ---------------------------
# Retrieval & prompt building
# ---------------------------

# --- Hybrid search: vector + keyword ---
def keyword_search(collection, query, n_results=RETRIEVE_K):
    # Try to match query as substring in text or source metadata
    try:
        # ChromaDB 0.4.x/0.5.x: use where filter for metadata, fallback to text search
        # This is a simple filter; for more advanced, use external search or re-rank
        res = collection.query(
            query_texts=[query],
            n_results=n_results,
            where={"$or": [
                {"source": {"$contains": query}},
                {"text": {"$contains": query}}
            ]}
        )
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"source": (m or {}).get("source"), "text": t} for m, t in zip(metas, docs)]
    except Exception:
        # fallback: return empty
        return []

@functools.lru_cache(maxsize=128)
def _inner_retrieve(q: str):
    # 1) vector search
    w1 = WF.query(query_texts=[q], n_results=RETRIEVE_K)
    d1 = DOCS.query(query_texts=[q], n_results=RETRIEVE_K)

    def pack(res):
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"source": (m or {}).get("source"), "text": t} for m, t in zip(metas, docs)]

    wf_hits = pack(w1)
    doc_hits = pack(d1)

    # 2) keyword search (hybrid)
    wf_kw = keyword_search(WF, q, n_results=RETRIEVE_K)
    doc_kw = keyword_search(DOCS, q, n_results=RETRIEVE_K)

    # 3) merge and deduplicate (by text)
    def dedup(items):
        seen = set()
        out = []
        for h in items:
            t = h["text"]
            if t and t not in seen:
                seen.add(t)
                out.append(h)
        return out

    wf_hits = dedup(wf_hits + wf_kw)
    doc_hits = dedup(doc_hits + doc_kw)

    # 4) widen if too few hits (vector expansion)
    if (len(wf_hits) + len(doc_hits)) < MIN_CONTEXT_HITS:
        widened = []
        for qq in expand_queries(q):
            r = WF.query(query_texts=[qq], n_results=RETRIEVE_K * 2)
            widened += pack(r)
        wf_hits = dedup(wf_hits + widened)

        widened = []
        for qq in expand_queries(q):
            r = DOCS.query(query_texts=[qq], n_results=RETRIEVE_K * 2)
            widened += pack(r)
        doc_hits = dedup(doc_hits + widened)

    # 5) enforce policy
    if (len(wf_hits) + len(doc_hits)) == 0 and REQUIRE_CONTEXT:
        raise HTTPException(status_code=422, detail="No matching context found. Refine your request or reindex new files.")

    return wf_hits, doc_hits

@langwatch.span(type="rag", name="retrieval", capture_input=True, capture_output=False)
def retrieve(q: str):
    wf_hits, doc_hits = _inner_retrieve(q)  # move your current body to a helper if you prefer

    # Attach contexts (trim if large)
    rag_contexts = []
    for i, item in enumerate((wf_hits[:6] + doc_hits[:6])):
        rag_contexts.append(
            RAGChunk(
                document_id=(item.get("source") or f"wfdoc-{i}"),
                content=item.get("text", ""),
            )
        )

    langwatch.get_current_span().update(
        contexts=rag_contexts,
        retrieval_strategy="chroma_vector_search",
        k=RETRIEVE_K,
    )
    return wf_hits, doc_hits



def pack_context(wf_chunks: List[Dict], doc_chunks: List[Dict], budget_tokens: int = 4000) -> str:
    parts, used = [], 0
    wi, di = 0, 0
    # take 2 workflow chunks then 1 doc chunk, repeat
    while used < budget_tokens and (wi < len(wf_chunks) or di < len(doc_chunks)):
        for _ in range(2):  # two WF
            if wi < len(wf_chunks):
                item = wf_chunks[wi]; wi += 1
                chunk = f"// {item.get('source')}\n{item.get('text')}\n"
                c = tcount(chunk)
                if used + c > budget_tokens: return "".join(parts)
                parts.append(chunk); used += c
        if di < len(doc_chunks):  # one DOC
            item = doc_chunks[di]; di += 1
            chunk = f"// {item.get('source')}\n{item.get('text')}\n"
            c = tcount(chunk)
            if used + c > budget_tokens: return "".join(parts)
            parts.append(chunk); used += c
    return "".join(parts)


# --- Flexible prompt builder ---
def build_prompt_flexible(prompt_input, context: str) -> str:
    # If prompt_input is a dict, use structured fields; else treat as plain text goal
    if isinstance(prompt_input, dict):
        goal = prompt_input.get("goal") or prompt_input.get("description") or ""
        triggers = prompt_input.get("triggers")
        integrations = prompt_input.get("integrations")
        steps = prompt_input.get("steps")
        constraints = prompt_input.get("constraints")
        extras = []
        if triggers:
            extras.append(f"TRIGGERS: {triggers}")
        if integrations:
            extras.append(f"INTEGRATIONS: {integrations}")
        if steps:
            extras.append(f"STEPS: {steps}")
        if constraints:
            extras.append(f"CONSTRAINTS: {constraints}")
        extras_str = "\n".join(extras)
        return f"""
Use the examples and docs below to produce a single **valid n8n workflow JSON** that satisfies the goal and requirements.

CONTEXT:
{context}

REQUIREMENTS:
- Output ONLY a single JSON object (no code fences, markdown, or commentary).
- Must include \"nodes\" and \"connections\" (and anything else n8n needs).
- Prefer official node names (e.g., HTTP Request, Slack, Airtable).
- Use placeholders for credentials; do not embed secrets.
- If you need clarification, include a single string field \"notes\" INSIDE the JSON with your question. Do NOT write anything outside the JSON.

{extras_str}

GOAL:
{goal}
""".strip()
    else:
        # fallback: treat as plain text goal
        return f"""
Use the examples and docs below to produce a single **valid n8n workflow JSON** that satisfies the goal.

CONTEXT:
{context}

REQUIREMENTS:
- Output ONLY a single JSON object (no code fences, markdown, or commentary).
- Must include \"nodes\" and \"connections\" (and anything else n8n needs).
- Prefer official node names (e.g., HTTP Request, Slack, Airtable).
- Use placeholders for credentials; do not embed secrets.
- If you need clarification, include a single string field \"notes\" INSIDE the JSON with your question. Do NOT write anything outside the JSON.

GOAL:
{prompt_input}
""".strip()


# ---------------------------
# JSON extraction
# ---------------------------
def extract_json(text: str) -> str:
    text = text.strip()
    # strip code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).strip()
        text = re.sub(r"\n?```$", "", text).strip()
    m = re.search(r"\{.*\}\s*$", text, flags=re.S)
    if not m:
        raise ValueError("No JSON object found in model output")
    return m.group(0)

# ---------------------------
# LLM call (OpenAI / Claude / Gemini)
# ---------------------------

# --- LLM output cache ---
_llm_cache = {}

@langwatch.span(type="llm", name="completion")
def llm_complete(prompt: str) -> str:
    # Use a hash of the prompt as the cache key
    key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if key in _llm_cache:
        return _llm_cache[key]

    provider = os.getenv("PROVIDER", "openai").lower()

    if provider == "openai":
        from openai import OpenAI
        oai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = [
            {"role": "system", "content": "You are an n8n workflow generator. Return ONLY a single valid JSON object."},
            {"role": "user", "content": prompt},
        ]

        r = oai.chat.completions.create(model=model, temperature=0.2, messages=messages)
        text = r.choices[0].message.content or ""

        langwatch.get_current_span().update(
            model=f"openai/{model}",
            input=messages,
            output=_truncate(text, 6000),
            metrics={
                "prompt_tokens": getattr(r, "usage", None) and r.usage.prompt_tokens or None,
                "completion_tokens": getattr(r, "usage", None) and r.usage.completion_tokens or None,
            },
        )
        _llm_cache[key] = text
        return text

    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set ANTHROPIC_API_KEY for anthropic provider.")
        client_a = anthropic.Anthropic(api_key=api_key)
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        msg = client_a.messages.create(
            model=model,
            temperature=0.2,
            max_tokens=4096,
            system="You are an n8n workflow generator. Return ONLY a single valid JSON object.",
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text if msg.content else ""
        langwatch.get_current_span().update(
            model=f"anthropic/{model}",
            input=[{"role":"system","content":"You are an n8n workflow generator. Return ONLY a single valid JSON object."},
                {"role":"user","content":prompt}],
            output=_truncate(text, 6000),
            metrics={
                "prompt_tokens": getattr(msg, "usage", None) and msg.usage.input_tokens or None,
                "completion_tokens": getattr(msg, "usage", None) and msg.usage.output_tokens or None,
            },
        )
        _llm_cache[key] = text
        return text

    elif provider == "gemini":
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GOOGLE_API_KEY for gemini provider.")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        model = genai.GenerativeModel(model_name)
        r = model.generate_content(prompt + "\n\nReturn ONLY a single JSON object.")
        text = getattr(r, "text", "") or ""
        langwatch.get_current_span().update(
            model=f"gemini/{model_name}",
            input=prompt,
            output=_truncate(text, 6000),
        )
        _llm_cache[key] = text
        return text

    _llm_cache[key] = ""
    return ""

# ---------------------------
# API models
# ---------------------------
class GenReq(BaseModel):
    prompt: str
    publish: bool = False

# ---------------------------
# Routes
# ---------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/search")
def search(q: str = Query(..., description="query"),
           k_wf: int = Query(4, ge=0, le=20),
           k_doc: int = Query(4, ge=0, le=20)):
    wf, dc = retrieve(q)
    wf = wf[:k_wf]
    dc = dc[:k_doc]
    return {
        "workflows": [{"source": w["source"], "preview": w["text"][:500]} for w in wf],
        "docs": [{"source": d["source"], "preview": d["text"][:500]} for d in dc],
    }

@app.post("/generate")
@langwatch.trace(name="generate_workflow", metadata={"service": "n8n-rag"})

def generate(req: GenReq):
    current = langwatch.get_current_trace()
    # Try to parse prompt as JSON for structured input, else fallback to plain text
    prompt_input = req.prompt
    parsed = None
    if isinstance(prompt_input, str):
        try:
            parsed = json.loads(prompt_input)
        except Exception:
            parsed = prompt_input
    else:
        parsed = prompt_input

    # --- Prompt rewriting step for plain text or ambiguous input ---
    rewritten = None
    if isinstance(parsed, dict):
        retrieval_query = parsed.get("goal") or parsed.get("description") or ""
        # If goal/description is short, rewrite it
        if retrieval_query and len(retrieval_query.split()) < 8:
            retrieval_query = rewrite_prompt(retrieval_query)
            parsed["goal"] = retrieval_query
    else:
        # If plain text and short, rewrite
        retrieval_query = str(parsed)
        if retrieval_query and len(retrieval_query.split()) < 8:
            retrieval_query = rewrite_prompt(retrieval_query)
            parsed = retrieval_query

    current.update(
        input=prompt_input,
        metadata={
            "retrieve_k": RETRIEVE_K,
            "min_context_hits": MIN_CONTEXT_HITS,
            "require_context": REQUIRE_CONTEXT,
            "query_expansion": QUERY_EXPANSION,
            "provider": os.getenv("PROVIDER", "openai"),
            "openai_model": os.getenv("OPENAI_MODEL"),
            "anthropic_model": os.getenv("ANTHROPIC_MODEL"),
            "gemini_model": os.getenv("GEMINI_MODEL"),
        },
    )
    # 1) retrieve
    wf_ctx, doc_ctx = retrieve(retrieval_query)
    context = pack_context(wf_ctx, doc_ctx, budget_tokens=4000)

    # 2) prompt + call model
    prompt = build_prompt_flexible(parsed, context)
    out = llm_complete(prompt)

    # 3) parse / validate
    try:
        wf_json = json.loads(extract_json(out))
    except Exception as e:
        current.update(output="error:invalid_json")
        return {"error": "Model did not return valid JSON", "details": str(e), "raw": out[:2000]}

    if not isinstance(wf_json, dict) or "nodes" not in wf_json or "connections" not in wf_json:
        current.update(output="error:missing_fields")
        return {"error": "Generated JSON missing 'nodes' or 'connections'", "raw": wf_json}

    current.update(output="workflow_json_ok")
    return {"workflow": wf_json}


# ---------------------------
# UI mount + UI-friendly endpoint
# ---------------------------
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")
@app.post("/ui_generate")
def ui_generate(
    req: GenReq,
    x_provider: str = Header(default=None),
    x_api_key: str = Header(default=None),     # keep param for compatibility
    x_model: str = Header(default=None),
    x_prefer_latest: str = Header(default="false"),
):
    # Allow UI to pick provider/model (no secrets)
    if x_provider:
        os.environ["PROVIDER"] = x_provider

    # Only accept keys from UI if explicitly allowed
    if ALLOW_UI_KEYS:
        if x_provider == "openai" and x_api_key:
            os.environ["OPENAI_API_KEY"] = x_api_key
        if x_provider == "anthropic" and x_api_key:
            os.environ["ANTHROPIC_API_KEY"] = x_api_key
        if x_provider == "gemini" and x_api_key:
            os.environ["GOOGLE_API_KEY"] = x_api_key

    if x_model:
        if x_provider == "openai":   os.environ["OPENAI_MODEL"]    = x_model
        if x_provider == "anthropic":os.environ["ANTHROPIC_MODEL"] = x_model
        if x_provider == "gemini":   os.environ["GEMINI_MODEL"]    = x_model

    prefer_latest = str(x_prefer_latest).lower() in ("1","true","yes")
    return generate(req)  # or pass prefer_latest through if you use it in generate()

def _reindex_worker():
    # Rebuild chunks
    subprocess.run([sys.executable, "chunk_all.py"], check=True)
    # Pick cloud vs local indexer automatically
    script = "build_chroma_cloud.py" if os.getenv("CHROMA_TOKEN") else "build_chroma.py"
    subprocess.run([sys.executable, script], check=True)

@app.post("/reindex")
def reindex(background: BackgroundTasks, x_admin_token: str = Header(None)):
    if x_admin_token != os.getenv("ADMIN_TOKEN"):
        raise HTTPException(status_code=401, detail="Unauthorized")
    background.add_task(_reindex_worker)
    return {"status": "started"}

@app.on_event("startup")
def _check_keys():
    provider = os.getenv("PROVIDER", "openai").lower()
    if provider == "openai" and not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_BASE_URL")):
        raise RuntimeError("Missing OPENAI_API_KEY (or OPENAI_BASE_URL for Ollama).")
    if provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("Missing ANTHROPIC_API_KEY.")
    if provider == "gemini" and not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY.")

# --- Prompt rewriting/expansion using LLM ---
def rewrite_prompt(raw_prompt: str) -> str:
    """
    Use the LLM to clarify and expand short or ambiguous prompts into actionable workflow instructions.
    """
    system_msg = (
        "You are an expert workflow designer. Rewrite the user's request as a clear, actionable workflow goal for an automation system. "
        "Be specific about triggers, steps, and outputs if possible. Do not add commentary."
    )
    prompt = f"USER REQUEST: {raw_prompt}\n\nRewrite as a clear workflow goal:"
    provider = os.getenv("PROVIDER", "openai").lower()
    # Use the same LLM as for generation, but with a different system prompt
    if provider == "openai":
        from openai import OpenAI
        oai = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        r = oai.chat.completions.create(model=model, temperature=0.2, messages=messages)
        return r.choices[0].message.content.strip()
    elif provider == "anthropic":
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client_a = anthropic.Anthropic(api_key=api_key)
        model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20240620")
        msg = client_a.messages.create(
            model=model,
            temperature=0.2,
            max_tokens=512,
            system=system_msg,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip() if msg.content else raw_prompt
    elif provider == "gemini":
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        model = genai.GenerativeModel(model_name)
        r = model.generate_content(prompt + "\n\nRewrite as a clear workflow goal.")
        return getattr(r, "text", raw_prompt).strip()
    else:
        return raw_prompt