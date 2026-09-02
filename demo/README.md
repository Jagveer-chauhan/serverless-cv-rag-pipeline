# Demo

This folder contains screenshots and documentation demonstrating the live application.

## Live Application

🌐 **Frontend**: https://serverless-cv-rag-pipeline.onrender.com  
📖 **API Docs**: https://serverless-cv-rag-pipeline.onrender.com/api/v1/docs  

## Screenshots

> Screenshots showing the full user flow:

### 1. Upload View
Drag-and-drop multi-PDF uploader with:
- Per-file processing status badges (`queued → extracting → extracted → indexing → rag_ready`)
- Real-time SLA Waterfall showing all 8 stage timings
- End-to-end total processing time

### 2. CV Detail & Chat View
3-pane layout:
- **Left**: CV list sidebar with candidate name, status badge, upload date
- **Center**: SSE RAG Chat with streaming responses and citation pills
- **Right tabs**: JSON Inspector (syntax-highlighted) | HR Profile View (formatted CV)

### 3. HR Profile View
Clean human-readable CV layout including:
- Summary, Experience timeline, Skills matrix
- Education, Certifications
- Derived insights (years of experience, seniority, domain)
- Inferred signals (leadership, communication style)

### 4. SLA Waterfall
Per-stage timing breakdown showing:
- Each of the 8 pipeline stages
- Duration in milliseconds
- Whether p95 ≤ 5.0s SLA was met
- Cold-start vs warm-path indicator

## How to Run Demo

```bash
# 1. Clone and start backend
cd backend
pip install -r requirements.txt
cp .env.example .env   # Fill in your HF_API_KEY and SUPABASE_DB_URL
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 2. Start frontend
cd frontend
npm install
npm run dev

# 3. Open http://localhost:5173
# 4. Drag a PDF from samples/ folder onto the upload zone
# 5. Watch the SLA waterfall appear in real-time
# 6. Click the CV to open the chat → ask "What are this candidate's top skills?"
```

## Sample CVs

Three representative CVs are available in `../samples/`:
| File | Role | Expected JSON |
|---|---|---|
| `cv1_backend_architect.pdf` | Backend Architect | `cv1_backend_architect.json` |
| `cv2_ml_engineer.pdf` | ML Engineer | `cv2_ml_engineer.json` |
| `cv3_frontend_lead.pdf` | Frontend Lead | `cv3_frontend_lead.json` |
