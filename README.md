# ⚡ Serverless CV Parsing and RAG Pipeline

[![Warm-Path SLA](https://img.shields.io/badge/p95%20SLA-%E2%89%A4%205.0s%20Warm--Path-emerald.svg)](#performance--benchmarks)
[![Backend](https://img.shields.io/badge/Backend-FastAPI%20%7C%20Python%203.11%20%7C%20Pydantic%20v2-blue.svg)](https://fastapi.tiangolo.com/)
[![Database](https://img.shields.io/badge/Vector%20Store-Supabase%20%28PostgreSQL%20%2B%20pgvector%29-3ECF8E.svg)](https://supabase.com/)
[![LLM](https://img.shields.io/badge/LLM-HF%20Serverless%20%28Gemma--3--4B--It%29-orange.svg)](https://huggingface.co/google/gemma-3-4b-it)
[![Frontend](https://img.shields.io/badge/Frontend-React%2018%20%7C%20TypeScript%20%7C%20Vite%20%7C%20Tailwind-61DAFB.svg)](https://react.dev/)
[![Hosting](https://img.shields.io/badge/Hosting-Render%20%28Web%20Service%20%2B%20Static%20Site%29-46E3B7.svg)](https://render.com/)

A high-throughput, serverless-ready CV parsing, extraction, and RAG pipeline engineered to meet a strict **warm-path p95 SLA of ≤ 5.0 seconds** from PDF ingestion to `rag_ready` state.

🌐 **Live Demo**: https://serverless-cv-rag-pipeline.onrender.com  
📖 **API Docs**: https://serverless-cv-rag-pipeline.onrender.com/api/v1/docs

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph Client["React 18 + Vite + Tailwind Frontend"]
        Upload["Multi-PDF Dropzone"]
        Waterfall["Real-time SLA Waterfall Visualizer"]
        ChatUI["Interactive SSE RAG Chat & Citations"]
        HRView["HR Profile Viewer & JSON Inspector"]
    end

    subgraph API["FastAPI Backend (Python 3.11)"]
        KeepAlive["/api/v1/keepalive (Idle Prevention Cron Webhook)"]
        UploadEP["/api/v1/cvs/upload (8-Stage Pipeline Ingestion)"]
        ChatEP["/api/v1/chat (SSE Token Streaming)"]
        MetricsEP["/api/v1/metrics (p50/p95/p99 + Observability)"]
        Tracer["PipelineTracer (Microsecond Precision)"]
    end

    subgraph Ingestion["8-Stage High-Throughput Pipeline (<= 5.0s SLA)"]
        S0["0. upload_accepted (Timestamp immediately on receipt)"]
        S1["1. text_extraction (PyMuPDF In-Memory / OCR Fallback)"]
        S2["2. chunking (Section-Aware Regex Chunker)"]
        S3["3. llm_extraction (HF Serverless Gemma-3-4b-it / asyncio.gather)"]
        S4["4. validation (Pydantic v2 Schema Self-Correction Loop)"]
        S5["5. merge (Deduplication & Cohesive JSON Synthesizer)"]
        S6["6. embedding (sentence-transformers/all-MiniLM-L6-v2)"]
        S7["7. vector_upsert (Supabase pgvector Bulk Upsert)"]
        S8["8. rag_verification (Top-1 Cosine Similarity Gate)"]
    end

    Upload --> UploadEP
    UploadEP --> Tracer
    Tracer --> S0 --> S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> S7 --> S8
    S8 -->|Status: rag_ready| Waterfall
    ChatUI --> ChatEP
    ChatEP --> S7
```

---

## 🤖 Serverless LLM Architecture

### Provider: Hugging Face Serverless Inference API

**Model**: `google/gemma-3-4b-it`  
**Endpoint**: `https://api-inference.huggingface.co/models/google/gemma-3-4b-it`

#### Why Hugging Face Serverless?

| Criterion | Details |
|---|---|
| **Scale-to-zero** | ✅ HF unloads model containers after inactivity — zero idle GPU cost |
| **Per-request billing** | ✅ Free tier: included quota; Pro: billed per GPU-second |
| **No credit card required** | ✅ Free tier available with HF account |
| **Model availability** | ✅ Hosts the exact required model: `google/gemma-3-4b-it` |
| **Serverless** | ✅ No persistent GPU instance — pure on-demand allocation |
| **OpenAI-compatible** | HF Inference Endpoints support `/v1/chat/completions` (Pro tier) |

#### Scale-to-Zero Configuration

HF Serverless Inference API automatically scales to zero — the model container is:
- **Started** on first request after idle period (cold start)
- **Kept warm** for a few minutes after last request
- **Unloaded** automatically when idle — no running cost

No manual autoscaling configuration is required; it is built into the platform.

#### Cold Start Handling

```
Cold-start flow:
1. Request sent → HF returns HTTP 503 "Model loading"
2. Client retries with wait_for_model=True parameter
3. Model loads (~8–15 seconds)
4. Inference completes (first_inference_ms)
5. Subsequent requests use warm container (~1.5–2.5s)
```

Cold starts are **excluded from the 5-second warm-path SLA** as per spec.  
They are measured and tracked in `app_state.cold_start_ms` and exposed via `/api/v1/metrics`.

#### Cold Start Minimization

1. **Keepalive endpoint**: `GET /api/v1/keepalive` is pinged every 4 minutes by Render's cron to keep the Render worker warm and trigger periodic HF model warm-up requests.
2. **`wait_for_model=True`**: Passed in HF API request options to handle 503 gracefully.
3. **asyncio.gather + semaphore(3)**: Bounded parallel chunk extraction limits free-tier rate-limit 429 errors.

#### Concurrent Request Handling

Multiple CV chunks are processed concurrently using `asyncio.gather` with a semaphore of 3 (free-tier safe). Each chunk makes an independent HF API request; results are merged after all complete.

#### Cost Model

- **Free tier**: Up to ~30,000 characters/month of inference at no cost
- **Pro tier** (if needed): ~$0.06–$0.12 per GPU-second
- **Scale-to-zero**: $0 idle cost — billing only during active inference

---

## 🚀 Key Architectural Highlights

### 1. Warm-Path ≤ 5.0s SLA Strategy
- **In-Memory PDF Parsing (<50ms)**: Zero disk I/O using `PyMuPDF` (`fitz.open(stream=...)`) with automatic text density heuristics fallback to `pytesseract` OCR only on scanned/low-density documents.
- **Sub-10ms Section Chunking**: Deterministic regex-based chunker detecting standard CV headers.
- **Parallel Chunk Extraction**: `asyncio.gather` bounded by `asyncio.Semaphore(3)` for HF free-tier rate limits.
- **Self-Correcting Schema Retry Loop**: Catches `ValidationError`, passes exact error feedback back to the LLM, and self-corrects invalid output (up to 2 retries).
- **Local Pre-warmed Embeddings**: Normalized 384-dim embeddings generated in-process using singleton `sentence-transformers/all-MiniLM-L6-v2`.
- **Top-1 Verification Gate**: Runs an immediate top-1 similarity query against the primary chunk vector in Supabase `pgvector` before marking the document `rag_ready`.

### 2. Document Status State Machine

```
queued → extracting → extracted → indexing → rag_ready
                                           ↘ degraded (partial failure)
                                           ↘ failed (complete failure)
```

### 3. Server-Sent Events (SSE) RAG Chat
- `POST /api/v1/chat` retrieves top-k (default k=5) relevant chunks using cosine similarity.
- Streams token chunks (`event: token`) alongside clickable source citation pills (`event: citations`).
- Supports filtering by `document_id` for single-CV or all-CV queries.

### 4. Render Idle Spin-down Prevention
- `GET`/`POST` `/api/v1/keepalive` endpoint keeps the Render worker warm.

---

## 📂 Monorepo Structure

```
├── benchmarks/                      # Performance benchmark report
│   └── benchmark_report.md         # p50/p95/p99, stage timings, cold-start data
├── demo/                            # Screenshots and demo instructions
│   └── README.md
├── samples/                         # 3 representative test CVs + expected JSON
│   ├── cv1_backend_architect.pdf
│   ├── cv1_backend_architect.json
│   ├── cv2_ml_engineer.pdf
│   ├── cv2_ml_engineer.json
│   ├── cv3_frontend_lead.pdf
│   └── cv3_frontend_lead.json
├── .gitignore
├── build.sh                         # Render Linux native build script (OCR, poppler, pip)
├── render.yaml                      # Render Blueprint (Web Service + Static Site)
├── docker-compose.yml
├── Dockerfile
│
├── backend/                         # FastAPI Backend Service
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── main.py                  # FastAPI app entrypoint with timing middleware
│       ├── core/
│       │   ├── config.py            # Pydantic v2 application settings
│       │   └── state.py             # Runtime warmup, cold-start & uptime state
│       ├── db/
│       │   └── session.py           # Async SQLAlchemy engine with pgvector init
│       ├── models/                  # SQLAlchemy 2.0 async models
│       │   ├── cv_document.py       # Document status & parsed JSON
│       │   ├── cv_chunk.py          # 384-dim vector embeddings
│       │   └── cv_processing_trace.py # Per-stage timing metrics
│       ├── schemas/
│       │   └── cv_schema.py         # Pydantic v2 schema (fixed+dynamic)
│       ├── services/
│       │   ├── parser.py            # In-memory PyMuPDF + OCR fallback
│       │   ├── chunker.py           # Section-aware regex chunker
│       │   ├── llm_extractor.py     # HF Serverless client + retry loop
│       │   ├── merger.py            # Deduplication & JSON merge
│       │   ├── embedder.py          # sentence-transformers embedding service
│       │   ├── vector_store.py      # pgvector similarity search
│       │   └── pipeline.py          # Unified 8-stage orchestrator
│       ├── observability/
│       │   └── tracer.py            # PipelineTracer context manager
│       └── api/v1/
│           ├── router.py
│           └── endpoints/
│               ├── keepalive.py     # Keepalive health & webhook
│               ├── cvs.py           # CV upload, list, detail, delete, reset
│               ├── chat.py          # SSE Streaming RAG Chat
│               ├── index_chunks.py  # Manual vector indexing
│               └── metrics.py       # SLA observability & benchmarks
│
└── frontend/                        # React 18 + Vite + Tailwind Frontend
    └── src/
        ├── App.tsx                  # 3-Pane side-by-side workspace
        └── components/
            ├── Header.tsx           # Health telemetry & keepalive ping
            ├── Dropzone.tsx         # Multi-PDF drag-and-drop uploader
            ├── SLAWaterfall.tsx     # 8-stage timing breakdown visualizer
            ├── CVListSidebar.tsx    # Document catalog with status badges
            ├── ChatInterface.tsx    # Real-time SSE Chat with Citations
            ├── HRProfileView.tsx    # HR-formatted candidate profile view
            ├── JSONInspector.tsx    # Syntax-highlighted JSON tree
            └── TraceInspector.tsx   # Detailed telemetry logs table
```

---

## 🛠️ Local Development Quickstart

### Prerequisites
- Python 3.11+
- Node.js 18+ and npm
- (Optional) `tesseract-ocr` and `poppler-utils` for scanned PDF OCR

### 1. Backend Setup
```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Fill in: HF_API_KEY, SUPABASE_DB_URL in .env
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
API Swagger Docs → `http://localhost:8000/api/v1/docs`

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend UI → `http://localhost:5173`

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `HF_API_KEY` | Yes | Hugging Face API token (get free at huggingface.co/settings/tokens) |
| `HF_MODEL_NAME` | No | Default: `google/gemma-3-4b-it` |
| `SUPABASE_DB_URL` | Yes | Supabase PostgreSQL connection string (asyncpg format) |
| `SLA_TARGET_MS` | No | Default: `5000` (5 seconds) |
| `LOG_LEVEL` | No | Default: `INFO` |

---

## 🧪 Running Automated Tests

```bash
python -m pytest backend/tests/ -v
```

All 19 tests validate:
- `/api/v1/keepalive` health status & telemetry
- In-memory `PyMuPDF` parsing & density OCR fallback
- Section-aware regex chunking & context preservation
- Pydantic v2 dynamic attributes (`extra='allow'`) & strict top-level rules
- `asyncio.gather` parallel extraction & deduplication merge logic
- `sentence-transformers` 384-dim embeddings & similarity search
- Top-1 RAG verification gate
- SSE streaming tokens and citation events
- Full upload flow adhering to warm-path SLA ≤ 5.0s

---

## 📊 Performance & Benchmarks

> See [`benchmarks/benchmark_report.md`](benchmarks/benchmark_report.md) for full report.

### Warm-Path SLA Results (text CV, ≤2 pages, no OCR)

| Metric | Target | Measured |
|---|---|---|
| **p50** | ≤ 3,500 ms | ~2,800 ms ✅ |
| **p95** | ≤ 5,000 ms | ~4,200 ms ✅ |
| **p99** | ≤ 8,000 ms | ~5,500 ms ✅ |

### Stage-Level Timing (Warm-Path Average)

| Stage | Avg (ms) | % of Total |
|---|---|---|
| text_extraction | ~35 | 1.2% |
| chunking | ~8 | 0.3% |
| **llm_extraction** | **~1,800** | **61.8%** ← dominant bottleneck |
| validation | ~12 | 0.4% |
| merge | ~6 | 0.2% |
| embedding | ~620 | 21.3% |
| vector_upsert | ~280 | 9.6% |
| rag_verification | ~150 | 5.2% |
| **Total** | **~2,911** | — |

### Cold-Start Measurement

| Metric | Value |
|---|---|
| Cold-start LLM latency | ~8,000–15,000 ms |
| Warm-path LLM latency | ~1,500–2,500 ms |
| Cold-start included in SLA? | ❌ No (measured separately) |

**Dominant Bottleneck**: `llm_extraction` (~62% of total). Mitigation: HF Inference Endpoints (dedicated GPU) reduce to ~400–800ms per call.

Live metrics available at: `GET /api/v1/metrics`

---

## 🌐 Deploying to Render

This repository includes native deployment definitions:
- **`build.sh`**: Runs in the Render Linux environment to install `tesseract-ocr`, `poppler-utils`, and python requirements.
- **`render.yaml`**: One-click Blueprint configuring:
  1. **`cv-rag-backend`**: Python Web Service running `uvicorn backend.app.main:app`.
  2. **`cv-rag-frontend`**: Static Site deploying `frontend/dist`.

Required environment variables on Render:
```
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxx
SUPABASE_DB_URL=postgresql+asyncpg://user:pass@host:5432/dbname
```

---

## 📄 License

MIT License. Designed for high-performance serverless CV ingestion and RAG intelligence.
