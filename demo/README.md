# 🎥 Product Demo & UI Walkthrough

This document outlines the live demonstration, UI architecture, sample queries, and walkthrough flows for the **Serverless CV Parsing and RAG Pipeline**.

---

## 🌐 Live Application URLs

| Resource | URL |
|---|---|
| **Web Application (React 18 Frontend)** | [https://serverless-cv-rag-pipeline.onrender.com](https://serverless-cv-rag-pipeline.onrender.com) |
| **Interactive API Docs (FastAPI Swagger)** | [https://serverless-cv-rag-pipeline.onrender.com/api/v1/docs](https://serverless-cv-rag-pipeline.onrender.com/api/v1/docs) |
| **Alternative ReDoc** | [https://serverless-cv-rag-pipeline.onrender.com/redoc](https://serverless-cv-rag-pipeline.onrender.com/redoc) |
| **SLA Metrics & Observability** | [https://serverless-cv-rag-pipeline.onrender.com/api/v1/metrics](https://serverless-cv-rag-pipeline.onrender.com/api/v1/metrics) |

---

## 🖥️ User Interface Overview & Visual Flows

The user interface is structured into an intuitive 3-pane workstation designed for HR & recruiting workflows:

```
+---------------------------------------------------------------------------------------------------------+
|  [⚡ CV RAG Pipeline]        [Health: Online]  [HF Warm]  [Keepalive Ping]  [Upload CV Modal]           |
+------------------------------------+------------------------------------+-------------------------------+
|  📄 CV List Sidebar                |  💬 Interactive SSE RAG Chat       |  📑 Tabbed Inspector Pane     |
|                                    |                                    |  [ HR Profile | JSON | Traces]|
|  • John Doe (Backend Architect)    |  Assistant: Welcome! Ask me...     |                               |
|    Badge: [rag_ready] ~2.8s        |                                    |  Candidate: John Doe          |
|  • Alex Chen (ML Engineer)         |  User: How many years of Python?   |  Domain: Cloud / SaaS         |
|    Badge: [rag_ready] ~3.1s        |                                    |  Experience: 7 Years (Senior) |
|  • Sarah Jenkins (Frontend Lead)   |  Assistant: John Doe has 7 years   |  Skills Matrix:               |
|    Badge: [rag_ready] ~2.9s        |  of Python experience at Acme...   |  - Python, FastAPI, Docker    |
|                                    |  [Citation: #0 CONTACT_HEADER]     |  - Kubernetes, PostgreSQL     |
|                                    |  [Citation: #2 EXPERIENCE]         |  - Inferred: System Design    |
+------------------------------------+------------------------------------+-------------------------------+
```

---

## 📸 Key UI Views & Features

### 1. Multi-PDF Drag & Drop Uploader (`Dropzone.tsx`)
- Drag and drop single or multiple PDF/DOCX files simultaneously.
- Real-time stage badges: `queued` ➔ `extracting` ➔ `extracted` ➔ `indexing` ➔ `rag_ready`.
- Instant latency breakdown rendered via the **SLA Waterfall Visualizer**.

### 2. Real-Time SLA Waterfall Visualizer (`SLAWaterfall.tsx`)
- Visual horizontal bar chart of all **8 pipeline stages**:
  1. `text_extraction` (PyMuPDF in-memory buffer)
  2. `chunking` (Section-aware regex splitter)
  3. `llm_extraction` (Hugging Face Serverless Gemma-3-4b-it)
  4. `validation` (Pydantic v2 self-correction loop)
  5. `merge` (Chunk deduplication & unification)
  6. `embedding` (`sentence-transformers/all-MiniLM-L6-v2`)
  7. `vector_upsert` (Supabase `pgvector`)
  8. `rag_verification` (Top-1 Cosine similarity gate)
- Displays total time in milliseconds and whether the execution satisfied the **≤ 5.0s SLA**.

### 3. Interactive SSE RAG Chat (`ChatInterface.tsx`)
- Server-Sent Events (SSE) token streaming for instantaneous response rendering.
- Clickable citation pills linking answers directly to their originating chunk index and section.
- Scoped querying (query against a single selected candidate or across all ingested CVs).

### 4. Tabbed Candidate Inspector (`HRProfileView.tsx` & `JSONInspector.tsx`)
- **HR Profile View**: Beautifully formatted human-readable CV view with:
  - Executive summary
  - Seniority, domain, and experience badge metrics
  - Interactive work experience timeline with bullet achievements
  - Skills matrix categorized into explicit, inferred, and soft skills
  - Education and verified certifications
  - Inferred behavioural and leadership signals
- **JSON Inspector**: Raw syntax-highlighted, copyable JSON tree.
- **Trace Inspector**: Microsecond-precision stage audit table.

---

## 🧪 Sample RAG Queries to Try

Test these representative queries in the live chat interface:

1. **Experience Verification**:
   > *"How many years of Python experience does John Doe have?"*
2. **Skill Analysis**:
   > *"What vector databases and machine learning frameworks has Alex Chen worked with?"*
3. **Cross-Candidate Comparison**:
   > *"Which candidate has experience with Kubernetes and distributed microservices?"*
4. **Soft Skills & Leadership**:
   > *"What leadership signals and team management experience are demonstrated in Sarah Jenkins' CV?"*

---

## 🚀 Running the Demo Locally

```bash
# 1. Start Backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # Set HF_API_KEY and SUPABASE_DB_URL
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start Frontend
cd frontend
npm install
npm run dev

# 3. Open http://localhost:5173
# 4. Drag any PDF from the samples/ directory
# 5. Observe real-time 8-stage SLA waterfall (<5.0s)
# 6. Ask questions in the streaming RAG chat
```

---

## 📁 Sample CV Dataset

| Sample File | Target Role | Expected JSON Schema |
|---|---|---|
| `../samples/cv1_backend_architect.pdf` | Senior Backend Architect | `../samples/cv1_backend_architect.json` |
| `../samples/cv2_ml_engineer.pdf` | ML & RAG Engineer | `../samples/cv2_ml_engineer.json` |
| `../samples/cv3_frontend_lead.pdf` | Frontend Lead | `../samples/cv3_frontend_lead.json` |

