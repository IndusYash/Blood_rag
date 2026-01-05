"""
Offline RAG-powered Blood Report System
100% Local - No API Keys Required

This module integrates:
1. Local embeddings (sentence-transformers)
2. Local vector DB (ChromaDB)
3. Local LLM (FLAN-T5)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path

# Local modules (all offline)
from ocr.ocr_reader import OCRReader
from utils.file_handler import FileHandler
from utils.blood_report_analyzer import BloodReportAnalyzer
from utils.local_medical_qa import LocalMedicalQA
from rag.retriever import RAGRetriever
from config import Config


# Initialize Flask
app = Flask(__name__)
CORS(app)

# Storage
report_storage = {}

# Initialize components (all offline)
print("\n" + "="*60)
print("🚀 Initializing Offline Medical AI System")
print("="*60)

print("\n1️⃣ Loading OCR (Tesseract)...")
ocr_reader = OCRReader()

print("2️⃣ Loading Blood Analyzer (Rule-based)...")
blood_analyzer = BloodReportAnalyzer()

print("3️⃣ Loading RAG Retriever (Offline)...")
print("   - Embedding model: sentence-transformers (local)")
print("   - Vector DB: ChromaDB (local)")
rag_retriever = RAGRetriever()

print("4️⃣ Loading Local LLM (FLAN-T5)...")
print("   ⚠️ First run will download ~300MB model")
print("   💡 Set use_model=False in LocalMedicalQA to skip LLM")
local_llm = LocalMedicalQA(
    model_name="google/flan-t5-small",
    use_model=True  # Set to False for faster startup (keyword-only mode)
)

print("\n" + "="*60)
print("✅ All components loaded successfully!")
print("🔒 100% Offline - No API keys needed")
print("="*60 + "\n")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'mode': 'offline-rag',
        'components': {
            'ocr': 'tesseract',
            'analyzer': 'rule-based',
            'embeddings': 'sentence-transformers (local)',
            'vector_db': 'chromadb (local)',
            'llm': 'flan-t5-small (local)',
            'api_keys_required': False
        }
    })


@app.route('/api/upload', methods=['POST'])
def upload_report():
    """Upload and analyze blood report"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file.filename:
            return jsonify({'error': 'Empty filename'}), 400
        
        print(f"\n📄 Processing: {file.filename}")
        
        # Save file
        file_handler = FileHandler(Config.UPLOAD_FOLDER)
        file_path = file_handler.save_uploaded_file(file)
        file_id = Path(file_path).stem  # Get filename without extension as ID
        print(f"💾 Saved: {file_id}")
        
        # OCR extraction
        print("🔍 Extracting text...")
        extracted_text = ocr_reader.extract_text(file_path)
        print(f"✅ Extracted {len(extracted_text)} characters")
        
        # Blood analysis
        print("🩸 Analyzing blood values...")
        blood_analysis = blood_analyzer.analyze_report(extracted_text)
        
        # Store in memory
        report_storage[file_id] = {
            'text': extracted_text,
            'analysis': blood_analysis,
            'file_path': file_path,
            'filename': file.filename
        }
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'text_length': len(extracted_text),
            'analysis': blood_analysis
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_report():
    """
    Query blood report using OFFLINE RAG + Local LLM
    No API keys needed!
    """
    try:
        data = request.json
        
        if 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        file_id = data.get('file_id', '')
        
        print(f"\n💬 Query: {query}")
        
        # Check if report exists
        if not file_id or file_id not in report_storage:
            return jsonify({
                'success': True,
                'response': "Please upload a blood report first!",
                'sources': [],
                'mode': 'offline'
            })
        
        report_data = report_storage[file_id]
        blood_analysis = report_data.get('analysis', {})
        
        # Step 1: Retrieve relevant medical knowledge (offline RAG)
        print("🔍 Searching medical knowledge base...")
        rag_results = rag_retriever.retrieve(query, n_results=3)
        
        if rag_results.get('error'):
            print(f"⚠️ RAG not initialized: {rag_results['error']}")
            context = ""
            sources = []
        else:
            context = rag_results.get('context', '')
            sources = rag_results.get('sources', [])
            print(f"📚 Found {len(sources)} relevant sources")
        
        # Step 2: Prepare blood values for LLM
        blood_values = {}
        if blood_analysis and blood_analysis.get('success'):
            for result in blood_analysis.get('results', []):
                blood_values[result['test_name']] = {
                    'value': result['value'],
                    'unit': result['unit'],
                    'status': result['interpretation']['status']
                }
        
        # Step 3: Generate response using local LLM (offline)
        print("🤖 Generating response with local LLM...")
        if local_llm.model is not None:
            response = local_llm.generate_response(
                query=query,
                context=context,
                blood_values=blood_values if blood_values else None
            )
        else:
            # Fallback to keyword matching if LLM not loaded
            response = generate_keyword_response(query, blood_analysis)
        
        print(f"✅ Response generated")
        
        return jsonify({
            'success': True,
            'response': response,
            'sources': sources,
            'mode': 'offline-rag' if local_llm.model else 'keyword'
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_keyword_response(query: str, blood_analysis: dict) -> str:
    """Fallback keyword-based response (if LLM not loaded)"""
    query_lower = query.lower()
    
    # Summary request
    if any(word in query_lower for word in ['summary', 'overview', 'results']):
        if blood_analysis and blood_analysis.get('success'):
            response = f"📋 **Summary**\n\n"
            response += f"Tests: {blood_analysis['total_tests']}\n"
            response += f"Abnormal: {blood_analysis['abnormal_count']}\n"
            return response
    
    # Specific test keywords
    test_keywords = {
        'wbc': ['wbc', 'white blood'],
        'hemoglobin': ['hemoglobin', 'hb '],
        'glucose': ['glucose', 'sugar']
    }
    
    if blood_analysis and blood_analysis.get('success'):
        for test_name, keywords in test_keywords.items():
            if any(kw in query_lower for kw in keywords):
                for result in blood_analysis['results']:
                    if result['test_name'] == test_name:
                        return f"🩸 {result['display_name']}: {result['value']} {result['unit']} [{result['interpretation']['status']}]"
    
    return "I can help with your blood report. Try: 'Show summary' or 'What is my WBC?'"


if __name__ == '__main__':
    # Create directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    print("\n🚀 Starting Offline RAG Server")
    print("="*60)
    print("📍 URL: http://localhost:5000")
    print("🔒 Mode: 100% Offline (No API keys)")
    print("="*60)
    print("\n📝 Endpoints:")
    print("   GET  /api/health  - System status")
    print("   POST /api/upload  - Upload blood report")
    print("   POST /api/query   - Ask questions (RAG + LLM)")
    print("\n💡 First query may be slower (model warmup)")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
