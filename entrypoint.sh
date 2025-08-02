#!/usr/bin/env sh
set -e

echo "🔧 Chunking docs & workflows (RUN_INDEX_ON_START=${RUN_INDEX_ON_START:-true})"
if [ "${RUN_INDEX_ON_START:-true}" = "true" ]; then
  python chunk_all.py || true
  python build_chroma.py
fi

echo "🚀 Starting API on 0.0.0.0:8000"
exec uvicorn mcp_server:app --host 0.0.0.0 --port 8000
