# Offline RAG Setup Guide
## 100% Local Medical AI - No API Keys Required

This guide shows you how to run the blood report system with RAG completely offline using local models.

---

## 🎯 What You Get (All Offline)

- ✅ **OCR**: Tesseract (local)
- ✅ **Embeddings**: sentence-transformers (local, ~90MB)
- ✅ **Vector Database**: ChromaDB (local storage)
- ✅ **LLM**: FLAN-T5-small (local, ~300MB)
- ✅ **No internet needed** after initial download
- ✅ **No API keys**
- ✅ **Free forever**

---

## 📦 Installation

### Step 1: Install RAG Dependencies

```bash
cd backend
pip install -r requirements_rag.txt
```

This installs:
- `sentence-transformers` - Embedding model (offline)
- `chromadb` - Vector database (offline)
- `transformers` - LLM framework
- `torch` - PyTorch (CPU version)
- `PyPDF2` - PDF processing

### Step 2: Build Medical Knowledge Base (One-time)

```bash
# Index the medical PDFs into vector database
python rag/embeddings.py
```

This processes:
- `rag/docs/blood_test_info.pdf`
- `rag/docs/indian_health_guidelines.pdf`
- `rag/docs/radiology_terms.pdf`

Output: Creates `rag/vector_store/` with embeddings (~5MB)

**First run**: Downloads sentence-transformers model (~90MB)

---

## 🚀 Running the System

### Option A: With RAG + Local LLM (Smart Responses)

```bash
cd backend
python app_rag_offline.py
```

**First run**: Downloads FLAN-T5-small (~300MB) - happens automatically  
**Startup time**: 15-30 seconds (loading models)  
**Memory usage**: ~2GB RAM  
**Response time**: 2-5 seconds per query

### Option B: Keyword-only (Fast, No LLM)

Edit `app_rag_offline.py` line 52:
```python
local_llm = LocalMedicalQA(
    model_name="google/flan-t5-small",
    use_model=False  # Set to False
)
```

**Startup time**: <2 seconds  
**Memory usage**: ~500MB RAM  
**Response time**: <100ms per query

---

## 📊 Comparison: Current vs Offline RAG

### Current System (app_simple.py)
- ✅ Fast startup (<1s)
- ✅ Low memory (300MB)
- ✅ Instant responses
- ❌ Keyword matching only
- ❌ Can't explain "why"
- ❌ No medical knowledge

### Offline RAG (app_rag_offline.py)
- ⚠️ Slower startup (15-30s)
- ⚠️ More memory (2GB)
- ⚠️ 2-5s per response
- ✅ Natural language understanding
- ✅ Answers complex questions
- ✅ Cites medical sources
- ✅ Explains relationships

---

## 💬 Example Queries

### Current System (Keyword)
```
❌ "Why is my cholesterol high?"
   → No match, returns help text

✅ "What is my cholesterol?"
   → Cholesterol: 210 mg/dL [HIGH]
```

### Offline RAG System
```
✅ "Why is my cholesterol high?"
   → Your cholesterol is 210 mg/dL which is above normal 
     (<200 mg/dL). This can be caused by diet high in 
     saturated fats, lack of exercise, or genetic factors.
     Consider reducing fatty foods and exercising 30 min daily.
     [Sources: blood_test_info.pdf, indian_health_guidelines.pdf]

✅ "What does high WBC indicate?"
   → High white blood cell count (WBC) typically indicates 
     an infection, inflammation, or immune response. Your 
     WBC is 12,000 which is slightly elevated. Common causes 
     include bacterial infections, stress, or smoking.

✅ "How are glucose and HbA1c related?"
   → Glucose measures your blood sugar at a single point, 
     while HbA1c shows your average glucose over 2-3 months.
     Your fasting glucose of 115 mg/dL and HbA1c of 5.8% 
     suggest prediabetes. Both tests confirm elevated blood 
     sugar control.
```

---

## 🔧 Configuration

Edit `config.py`:

```python
# Embedding Model (offline)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 90MB

# LLM Options (offline)
LLM_MODEL = "google/flan-t5-small"  # 300MB, fast
# LLM_MODEL = "google/flan-t5-base"  # 1GB, better quality
# LLM_MODEL = "google/flan-t5-large"  # 3GB, best quality

# Vector Database
CHROMA_PERSIST_DIRECTORY = "rag/vector_store"
CHROMA_COLLECTION_NAME = "medical_knowledge"

# Retrieval
TOP_K_RESULTS = 3  # Number of chunks to retrieve
CHUNK_SIZE = 512   # Characters per chunk
CHUNK_OVERLAP = 50 # Overlap between chunks
```

---

## 📁 Directory Structure

```
backend/
├── app_rag_offline.py          # New: Offline RAG server
├── app_simple.py                # Old: Keyword-only server
├── rag/
│   ├── embeddings.py            # Build vector DB
│   ├── retriever.py             # Query vector DB
│   ├── docs/                    # Medical PDFs (add more here)
│   │   ├── blood_test_info.pdf
│   │   ├── indian_health_guidelines.pdf
│   │   └── radiology_terms.pdf
│   └── vector_store/            # Generated (ChromaDB data)
├── utils/
│   └── local_medical_qa.py      # Local FLAN-T5 LLM
└── requirements_rag.txt         # New dependencies
```

---

## 🧪 Testing

### Test 1: Check Health
```bash
curl http://localhost:5000/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "mode": "offline-rag",
  "components": {
    "ocr": "tesseract",
    "llm": "flan-t5-small (local)",
    "api_keys_required": false
  }
}
```

### Test 2: Test Local LLM
```bash
cd backend/utils
python local_medical_qa.py
```

### Test 3: Test RAG Retrieval
```bash
cd backend/rag
python retriever.py
```

---

## 🎓 Adding More Medical Knowledge

1. Add PDF files to `backend/rag/docs/`
2. Re-run indexing:
   ```bash
   python rag/embeddings.py
   ```
3. Restart server

**Recommended PDFs:**
- Blood test reference ranges
- Medical terminology glossary
- Disease symptom guides
- Medication information
- Dietary guidelines

---

## ⚡ Performance Tips

### For Faster Responses:
1. Use `flan-t5-small` (not base/large)
2. Reduce `TOP_K_RESULTS` to 2
3. Reduce `CHUNK_SIZE` to 256
4. Use GPU if available (auto-detected)

### For Better Quality:
1. Upgrade to `flan-t5-base` (1GB)
2. Increase `TOP_K_RESULTS` to 5
3. Add more medical PDFs
4. Use larger `CHUNK_SIZE` (1024)

---

## 🐛 Troubleshooting

### "Collection not found" Error
```bash
# Build vector database first
python rag/embeddings.py
```

### "Model not found" Error
```bash
# Check internet on first run (downloads model)
# Or manually download from HuggingFace
```

### High Memory Usage
```bash
# Use smaller model
LLM_MODEL = "google/flan-t5-small"

# Or disable LLM
use_model=False  # in app_rag_offline.py
```

### Slow Responses
```bash
# Reduce context
TOP_K_RESULTS = 2

# Reduce max response length
max_length=128  # in local_medical_qa.py
```

---

## 📊 System Requirements

### Minimum (Keyword Mode):
- CPU: Any x64
- RAM: 4GB
- Disk: 500MB

### Recommended (RAG + LLM):
- CPU: Modern x64 (4+ cores)
- RAM: 8GB
- Disk: 2GB
- GPU: Optional (10x faster)

---

## 🔄 Switching Between Modes

### Run Current System (Fast, Keyword-only):
```bash
python app_simple.py
```

### Run RAG System (Smart, Offline):
```bash
python app_rag_offline.py
```

Both run on port 5000 - just use one at a time!

---

## ✅ Summary

**Offline RAG gives you:**
- Natural language understanding
- Complex question answering
- Medical knowledge citations
- Contextual recommendations
- 100% privacy (no cloud)

**Trade-offs:**
- Slower startup
- More memory
- Slower responses

**Perfect for:**
- Medical Q&A chatbot
- Educational tool
- Privacy-sensitive applications
- No internet environments

**Not needed for:**
- Simple "what is my X value?" queries
- Fast lookup systems
- Low-resource devices
