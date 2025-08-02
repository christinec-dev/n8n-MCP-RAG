# build_chroma_cloud.py
import os, glob, json, hashlib
from typing import List
import tiktoken, chromadb

# --- connect to Chroma Cloud using your CLI creds (.env) ---
def env_bool(k, default="true"):
    return str(os.getenv(k, default)).lower() in ("1", "true", "yes")

client = chromadb.HttpClient(
    ssl=env_bool("CHROMA_SSL", "true"),
    host=os.getenv("CHROMA_HOST", "api.trychroma.com"),
    port=int(os.getenv("CHROMA_PORT", "8000")),
    tenant=os.getenv("CHROMA_TENANT"),
    database=os.getenv("CHROMA_DATABASE", "n8n"),
    headers={"x-chroma-token": os.getenv("CHROMA_TOKEN")},
)

def recoll(name: str):
    try: client.delete_collection(name)
    except Exception: pass
    return client.get_or_create_collection(name)

wf   = recoll("n8n-workflows")
docs = recoll("n8n-docs")

# --- embeddings (OpenAI or local fallback) ---
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    from openai import OpenAI
    oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    def embed_many(texts: List[str]):
        out = oai.embeddings.create(model="text-embedding-3-small", input=texts)
        return [d.embedding for d in out.data]
else:
    from sentence_transformers import SentenceTransformer
    _m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    def embed_many(texts: List[str]):
        return _m.encode(texts, normalize_embeddings=True).tolist()

# --- splitting / batching (token-safe) ---
enc = tiktoken.get_encoding("cl100k_base")
MAX_SEG = int(os.getenv("EMBED_MAX_TOKENS", "7500"))
OVERLAP = int(os.getenv("EMBED_OVERLAP", "100"))
MAX_REQ_TOKENS = int(os.getenv("EMBED_MAX_REQUEST_TOKENS", "250000"))
MAX_REQ_ITEMS  = int(os.getenv("EMBED_MAX_REQUEST_ITEMS", "128"))

def split_for_embedding(text: str):
    ids = enc.encode(text)
    if len(ids) <= MAX_SEG: return [text]
    parts, start = [], 0
    while start < len(ids):
        end = min(start + MAX_SEG, len(ids))
        parts.append(enc.decode(ids[start:end]))
        if end == len(ids): break
        start = max(0, end - OVERLAP)
    return parts

def clean_meta(meta: dict):
    def norm(v):
        if v is None: return None
        if isinstance(v, (str,int,float,bool)): return v
        return str(v)
    return {k: norm(v) for k,v in meta.items() if k != "content"}

def load_segments(pattern: str, kind: str):
    items = []
    for p in glob.glob(pattern, recursive=True):
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
        base = clean_meta({k:v for k,v in obj.items() if k!="content"})
        segs = split_for_embedding(obj["content"])
        total = len(segs)
        for i, seg in enumerate(segs, 1):
            meta = dict(base)
            meta.update({"type": kind, "source": base.get("source"),
                         "segment": i, "segments": total})
            pid = hashlib.sha1(f"{p}#seg{i:03d}".encode()).hexdigest()
            items.append((seg, meta, pid, len(enc.encode(seg))))
    return items

def upsert(collection, items):
    i, n, batch_no = 0, len(items), 1
    while i < n:
        tok_sum = 0; batch = []
        while i < n and len(batch) < MAX_REQ_ITEMS:
            text, meta, pid, tlen = items[i]
            if batch and (tok_sum + tlen > MAX_REQ_TOKENS): break
            batch.append((text, meta, pid)); tok_sum += tlen; i += 1
        texts = [b[0] for b in batch]
        vecs  = embed_many(texts)
        collection.add(
            documents=texts,
            metadatas=[b[1] for b in batch],
            ids=[b[2] for b in batch],
            embeddings=vecs
        )
        print(f"Upserted {len(batch)} (batch {batch_no})"); batch_no += 1

if __name__ == "__main__":
    wf_items  = load_segments("data/chunks/workflows/*.json", "workflow")
    doc_items = load_segments("data/chunks/docs/**/*.json", "doc")
    print(f"Prepared {len(wf_items)} workflow segs, {len(doc_items)} doc segs")
    upsert(wf,  wf_items)
    upsert(docs, doc_items)
    print("✅ Uploaded to Chroma Cloud")
