# n8n Workflow Builder (RAG)

This tool generates importable n8n workflows from plain English descriptions, using Retrieval-Augmented Generation (RAG) to reduce hallucinations and match your style. It supports OpenAI, Anthropic, Google Gemini, and Ollama/LM Studio via OpenAI-compatible endpoints.

## Key Features
- **RAG Implementation**: Retrieves relevant chunks from indexed docs/workflows before generating.
- **FastAPI Backend**: Handles API requests, retrieval, and workflow generation.
- **Static UI**: Simple web interface at `/ui` for user interaction.
- **Flexible Storage**: Uses local Chroma DB or Chroma Cloud for vector storage.
- **Provider Support**: OpenAI, Anthropic, Gemini, and OpenAI-compatible endpoints.

## Example:
<img width="1318" height="673" alt="image" src="https://github.com/user-attachments/assets/d7227c04-f2fe-4805-8627-c5ee006318be" />
<img width="1329" height="640" alt="image" src="https://github.com/user-attachments/assets/33f7d754-ad67-4ded-9390-7c1e39d757c1" />
<img width="956" height="271" alt="image" src="https://github.com/user-attachments/assets/793db97b-c184-4fcf-90a3-44a5d19fc89d" />

## Project Structure
```
.
├─ mcp_server.py          # FastAPI app (endpoints, retrieval, prompt building, model calling)
├─ chunk_all.py           # Splits docs/workflows into JSON chunks
├─ build_chroma.py        # Embeds chunks into local Chroma DB
├─ build_chroma_cloud.py  # Embeds chunks into Chroma Cloud (optional)
├─ ui/                    # Static UI files (served at /ui)
├─ data/                  # Directory for chunked data
│  └─ chunks/             # Contains generated chunks
├─ chroma_db/             # Local Chroma DB (not checked in)
├─ Dockerfile
├─ docker-compose.yml
└─ .env                   # Configuration file (not committed)
```

## Requirements
- Docker and Docker Compose
- API key for one of the supported providers (e.g., OpenAI, Anthropic, etc.)
- (Optional) A Chroma Cloud account/token if you prefer cloud indexing.

## Configuration
Create a `.env` file in the project root with your settings. Example configurations:

### A) OpenAI Provider
```env
PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-api-key  # Or leave empty to input via UI
```

### B) Ollama Provider
```env
PROVIDER=openai
OPENAI_MODEL=llama3.1:8b
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
OPENAI_API_KEY=ollama  # Dummy key for Ollama compatibility
```

### C) Local Chroma DB
If using local Chroma, set:
```env
RUN_INDEX_ON_START=false
```

## Running the Project
### Using Docker
1. Build and start the container:
   ```bash
   docker compose up -d --build
   docker compose logs -f app
   ```
   The app runs on `http://localhost:8000` by default.

2. Access the UI at `http://localhost:8000/ui`.

### Indexing Your Data
- **One-Time Local Indexing**:
  ```bash
  docker compose run --rm app python chunk_all.py
  docker compose run --rm app python build_chroma.py
  ```

- **Reindex on Demand**:
  ```bash
  curl -X POST "http://localhost:8000/reindex" -H "X-Admin-Token: your-admin-token"
  ```

## Using the Application
- **UI Interaction**: Describe your workflow in plain English, select a provider and model, and generate the workflow JSON.
- **Endpoints**:
  - `GET /health`: Check server status.
  - `GET /search?q=...`: Retrieve context snippets (for debugging).
  - `POST /generate`: Submit a prompt and get the generated workflow JSON.
  - `POST /ui_generate`: Same as `/generate` but accepts provider/model/API key headers.

## Troubleshooting
- **Model Errors**: Use a stronger model (e.g., `gpt-4o` or `claude-3-5-sonnet`) or clarify your prompt.
- **Indexing Issues**: Ensure your Chroma token and settings are correct. Stop the container and rebuild if needed.
- **Docker Problems**: Verify Docker is running and network settings are correct.

## Security Notes
- API keys are sent per request via headers and not stored on the server.
- Protect sensitive endpoints with strong tokens and consider using a reverse proxy for production.

## License
MIT (Add your preferred license here)

## Quick Start
1. Clone the repository and copy the `.env.example` to `.env`.
2. (Optional) Run indexing:
   ```bash
   docker compose run --rm app python chunk_all.py
   docker compose run --rm app python build_chroma.py
   ```
3. Start the app:
   ```bash
   docker compose up -d --build
   docker compose logs -f app
   ```
4. Open `http://localhost:8000/ui` and start generating workflows!
