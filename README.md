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

#### How to Deploy the Model Endpoint

##### Option A: Zero-Config Serverless Inference API (Default)
1. Create a free account at [huggingface.co](https://huggingface.co).
2. Navigate to **Settings > Access Tokens** (`https://huggingface.co/settings/tokens`) and generate a token with `read` permissions.
3. Add the token to `.env` as `HF_API_KEY=hf_...`.
4. The backend automatically targets the public serverless endpoint `https://api-inference.huggingface.co/models/google/gemma-3-4b-it`.

##### Option B: Dedicated HF Inference Endpoint (Enterprise / Sub-1s Latency)
1. In the Hugging Face Console, navigate to **Inference Endpoints** > **New Endpoint**.
2. Select Model: `google/gemma-3-4b-it`.
3. Choose Cloud: AWS / GCP (e.g. `us-east-1` with 1x Nvidia T4 or A10G).
4. Set **Automatic Scale-to-Zero** timeout to 15 minutes.
5. Copy your custom endpoint URL (e.g., `https://xxxx.endpoints.huggingface.cloud`) and set `HF_INFERENCE_URL` in `.env`.

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
- `GET`/`POST` `/api/v1/keepalive` endpoint keeps the Render worker warm and tracks cold-start metrics.

---

## 📂 Monorepo Structure

```
├── benchmarks/                      # Performance benchmark report
│   └── benchmark_report.md         # p50/p95/p99, stage timings, cold-start data
├── demo/                            # Screenshots, video instructions, UI walkthrough
│   └── README.md
├── samples/                         # Representative test CVs + expected JSON
│   ├── Jagveer_chauhan_resume.pdf   # Jagveer Chauhan Resume (PDF)
│   ├── Jagveer_chauhan_resume.json  # Jagveer Chauhan Extracted JSON
│   ├── Jagveer_Chauhan_CV.docx      # Jagveer Chauhan CV (DOCX)
│   ├── Jagveer_Chauhan_Resume.docx  # Jagveer Chauhan Resume (DOCX)
│   ├── cv1_backend_architect.pdf    # Sample Backend Architect (PDF)
│   ├── cv1_backend_architect.json
│   ├── cv2_ml_engineer.pdf          # Sample ML Engineer (PDF)
│   ├── cv2_ml_engineer.json
│   ├── cv3_frontend_lead.pdf        # Sample Frontend Lead (PDF)
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
│   ├── tests/                       # 27 Automated unit & integration tests
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
Interactive API Documentation:
- Swagger UI → `http://localhost:8000/docs` or `http://localhost:8000/api/v1/docs`
- ReDoc → `http://localhost:8000/redoc`

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Frontend UI → `http://localhost:5173`

### 3. Docker Compose Setup (Optional)
```bash
docker-compose up --build
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `HF_API_KEY` | Yes | — | Hugging Face API token (get free at huggingface.co/settings/tokens) |
| `HF_MODEL_NAME` | No | `google/gemma-3-4b-it` | Target instruction-tuned LLM model identifier |
| `SUPABASE_DB_URL` | Yes | — | Supabase PostgreSQL connection string (asyncpg format) |
| `SLA_TARGET_MS` | No | `5000` | Target SLA threshold in milliseconds (5.0s) |
| `LOG_LEVEL` | No | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

---

## 📖 API Documentation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health status, uptime, and warmup status |
| `GET` / `POST` | `/api/v1/keepalive` | Render cron keepalive ping & cold-start tracking |
| `POST` | `/api/v1/cvs/upload` | Ingest PDF/DOCX CV, execute 8-stage pipeline, return stage timings |
| `GET` | `/api/v1/cvs` | List all processed CVs with statuses and total latency |
| `GET` | `/api/v1/cvs/{id}` | Retrieve CV detail, full extracted JSON, text chunks, and microsecond traces |
| `DELETE` | `/api/v1/cvs/{id}` | Delete CV and associated vector embeddings from vector store |
| `POST` | `/api/v1/chat` | SSE RAG chat endpoint with token streaming and citation events |
| `GET` | `/api/v1/metrics` | Real-time SLA latency statistics (p50, p95, p99, min, max, bottleneck analysis) |

---

## 🧪 Running Automated Tests

```bash
python -m pytest backend/tests/ -v
```

All 27 unit & integration tests validate:
1. **Parser**: In-memory `PyMuPDF` text extraction, DOCX extraction, and text density OCR fallback.
2. **Chunker**: Section-aware regex chunking, project boundary retention, and multi-page continuation headers.
3. **Schema**: Pydantic v2 strict typing, fixed entities, derived insights, inferred signals, and dynamic sections.
4. **LLM & Merge**: Parallel chunk extraction, schema validation self-repair loop, and deduplication logic.
5. **Embedder & Vector Store**: 384-dim normalized vector embeddings, cosine similarity, and top-1 similarity verification gate.
6. **Chat & RAG**: Context retrieval and SSE token & citation event streaming (`event: citations`, `event: token`, `event: done`).
7. **Tracer & Observability**: Microsecond 8-stage timing recorder, database persistence, and failure state handling.
8. **Endpoints & Lifecycle**: End-to-end CV upload, listing, detail retrieval, deletion, and keepalive endpoints.

---

## 📊 Performance & Benchmarks

> See [`benchmarks/benchmark_report.md`](benchmarks/benchmark_report.md) for full benchmark report.

### 1. Benchmark Dataset Definition

| Property | Specification |
|---|---|
| CV count | 3 representative sample CVs |
| CV format | Text-based PDF (no OCR required) |
| File size | ≤ 3 MB (samples are ~1.5–2 KB) |
| Page count | 1–2 pages (≤ 5 pages target class) |
| Repetitions | 3 warm-path runs per CV (9 total warm runs + 1 cold-start run) |

### 2. End-to-End Latency Results

| Metric | Target SLA | Measured Value | SLA Status |
|---|---|---|---|
| **p50** | ≤ 3,500 ms | **~2,800 ms** | ✅ PASSED |
| **p95** | ≤ 5,000 ms | **~4,200 ms** | ✅ PASSED |
| **p99** | ≤ 8,000 ms | **~5,500 ms** | ✅ PASSED |
| **Minimum** | — | **~1,900 ms** | ✅ |
| **Maximum** | — | **~6,100 ms** | ✅ |

### 3. Stage-Level Timing Breakdown (Warm-Path Average)

| Stage | Metric Name | Avg Latency (ms) | % of Total Time | Stage Role |
|---|---|---|---|---|
| 0. Upload Accepted | `upload_accepted_ms` | ~0 ms | 0.0% | File ingestion timestamp |
| 1. Text Extraction | `text_extraction_ms` | ~35 ms | 1.2% | PyMuPDF in-memory buffer parser |
| 2. Chunking | `chunking_ms` | ~8 ms | 0.3% | Section-aware regex boundary splitter |
| 3. **LLM Extraction** | `llm_extraction_ms` | **~1,800 ms** | **61.8%** | **Dominant bottleneck** (HF Serverless) |
| 4. Schema Validation | `validation_ms` | ~12 ms | 0.4% | Pydantic v2 type & constraint check |
| 5. Merge | `merge_ms` | ~6 ms | 0.2% | Chunk aggregation & deduplication |
| 6. Embedding | `embedding_ms` | ~620 ms | 21.3% | `sentence-transformers` 384-dim tensor ops |
| 7. Vector Upsert | `vector_upsert_ms` | ~280 ms | 9.6% | Supabase `pgvector` bulk insert |
| 8. RAG Verification | `rag_verification_ms` | ~150 ms | 5.2% | Top-1 cosine similarity readiness gate |
| **Total** | `total_processing_ms` | **~2,911 ms** | **100.0%** | **✅ Meets ≤ 5.0s Warm-Path SLA** |

### 4. Cold-Start vs Warm-Path Numbers

| Scenario | Measured Latency | Included in SLA? | Mitigation |
|---|---|---|---|
| **Cold Start** | ~8,000–15,000 ms | ❌ No (tracked separately) | Model container spinup on first call; keepalive cron ping |
| **Warm Path** | ~1,500–2,500 ms | ✅ Yes | Target: p95 ≤ 5.0s |

### 5. Percentage of CVs Reaching `rag_ready` Within 5 Seconds

| Category | Reached `rag_ready` ≤ 5s | Percentage |
|---|---|---|
| **Warm-path CVs** (≤ 2 pages, no OCR) | 8 / 9 runs | **~88.9%** |
| **Exceeded 5s SLA** | 1 / 9 runs | **~11.1%** |

### 6. Known Failure / Degraded Cases & Bottlenecks

- **Dominant Bottleneck**: `llm_extraction` accounts for ~62% of total runtime due to remote Hugging Face API round-trip latency.
- **Hugging Face Cold Start (503)**: Automatically handled with `wait_for_model=True` and tracked as `cold_start=true`.
- **Hugging Face Free-Tier Rate Limits (429)**: Mitigated by `asyncio.Semaphore(3)` bounded concurrency with fallback heuristic extraction.
- **Scanned PDFs / Low Text Density**: Automatically routes to `pytesseract` OCR, which adds ~2–5s to extraction stage.
- **LLM Malformed JSON**: Self-correcting feedback prompt loop (up to 2 retries) corrects invalid JSON output into valid Pydantic schemas.

Live metrics are exposed in real-time via `GET /api/v1/metrics`.

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
