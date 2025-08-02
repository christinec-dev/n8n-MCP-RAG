import re, json
from pathlib import Path
from typing import List
import tiktoken

DOCS_ROOT = Path("data/docs/docs")
WF_ROOT   = Path("data/workflows")
OUT_DOCS  = Path("data/chunks/docs")
OUT_WF    = Path("data/chunks/workflows")

# token target & overlap
CHUNK_TOKENS = 800
OVERLAP_TOKENS = 120
enc = tiktoken.get_encoding("cl100k_base")

def tok_count(text: str) -> int:
    return len(enc.encode(text))

def split_by_headers(md: str) -> List[str]:
    # keep headers as part of each section
    parts = re.split(r'(?=^#{1,3}\s)', md, flags=re.M)
    return [p for p in parts if p.strip()]

def pack_tokens(text: str, chunk_tokens=CHUNK_TOKENS, overlap=OVERLAP_TOKENS) -> List[str]:
    ids = enc.encode(text)
    out, start = [], 0
    while start < len(ids):
        end = min(start + chunk_tokens, len(ids))
        chunk_ids = ids[start:end]
        out.append(enc.decode(chunk_ids))
        if end == len(ids): break
        start = max(0, end - overlap)
    return out

def chunk_markdown_file(path: Path) -> List[dict]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    sections = split_by_headers(raw) or [raw]
    idx, chunks = 1, []
    for sec in sections:
        for piece in pack_tokens(sec):
            chunks.append({
                "source": str(path.relative_to(DOCS_ROOT)),
                "index": idx,
                "content": piece.strip(),
            })
            idx += 1
    return chunks

def dump_json(obj: dict, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def chunk_all_docs():
    print("Chunking docs …")
    for md in DOCS_ROOT.rglob("*.md"):
        out_dir = OUT_DOCS / md.parent.relative_to(DOCS_ROOT)
        chunks = chunk_markdown_file(md)
        for c in chunks:
            out_path = out_dir / f"{md.stem}__{c['index']:03}.json"
            dump_json(c, out_path)
    print("✅ Docs chunked ->", OUT_DOCS)

def extract_workflow_metadata(raw: str):
    try:
        data = json.loads(raw)
        nodes = data.get("nodes", [])
        node_types, trigger = [], None
        for n in nodes:
            t = n.get("type")
            if t: node_types.append(str(t))
            nm = (n.get("name") or "").lower()
            if "trigger" in nm and not trigger:
                trigger = nm
        return {"node_types": node_types[:60], "trigger": trigger}
    except Exception:
        return {}

def chunk_all_workflows():
    print("Writing one chunk per workflow …")
    for wf in sorted(WF_ROOT.glob("*.json")):
        raw = wf.read_text(encoding="utf-8", errors="ignore")
        meta = extract_workflow_metadata(raw)
        doc = {"source": wf.name, "index": 1, "content": raw.strip(), "metadata": meta}
        out_path = OUT_WF / f"{wf.stem}__001.json"
        dump_json(doc, out_path)
    print("✅ Workflows chunked ->", OUT_WF)

if __name__ == "__main__":
    OUT_DOCS.mkdir(parents=True, exist_ok=True)
    OUT_WF.mkdir(parents=True, exist_ok=True)
    chunk_all_docs()
    chunk_all_workflows()