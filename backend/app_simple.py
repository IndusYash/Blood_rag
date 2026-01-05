"""
Medical Report AI - SIMPLIFIED VERSION
Focus: Blood Report OCR + Analysis + Q&A
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from config import Config
from ocr.ocr_reader import OCRReader
from utils.file_handler import FileHandler
from utils.blood_report_analyzer import BloodReportAnalyzer
from utils.local_medical_qa import LocalMedicalQA
from rag.retriever import RAGRetriever

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize components
print("\n" + "="*60)
print("🏥 MEDICAL REPORT AI - BLOOD REPORT ANALYZER")
print("="*60)
print("Initializing components...")

ocr_reader = OCRReader()
file_handler = FileHandler()
blood_analyzer = BloodReportAnalyzer()

# Initialize RAG components (offline)
print("Loading RAG + Local LLM (offline)...")
rag_retriever = RAGRetriever()
local_llm = LocalMedicalQA(
    model_name="google/flan-t5-small",
    use_model=True  # Offline LLM enabled
)

print("✅ System Ready with Offline RAG!")
print("="*60 + "\n")

# Store reports in memory
report_storage = {}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Blood Report AI is running',
        'version': '2.0-simplified'
    })


@app.route('/api/upload', methods=['POST'])
def upload_report():
    """
    Upload and analyze blood report
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save file
        file_path = file_handler.save_uploaded_file(file)
        print(f"\n📁 File saved: {file_path}")
        
        # Extract text using OCR
        print(f"🔍 Extracting text...")
        extracted_text = ocr_reader.extract_text(file_path)
        
        if not extracted_text or len(extracted_text) < 10:
            return jsonify({
                'error': 'Could not extract text from report',
                'suggestion': 'Please ensure: 1) Image is clear and high quality, 2) Text is readable, 3) File is not corrupted',
                'extracted_text': extracted_text,
                'text_length': len(extracted_text)
            }), 400
        
        print(f"✅ Extracted {len(extracted_text)} characters")
        
        # Generate file ID
        file_id = Path(file_path).stem
        
        # Analyze blood report
        print(f"🩸 Analyzing blood values...")
        blood_analysis = blood_analyzer.analyze_report(extracted_text, gender='normal')
        
        # Store in memory
        report_storage[file_id] = {
            'text': extracted_text,
            'file_path': file_path,
            'analysis': blood_analysis
        }
        
        print(f"✅ Analysis complete!\n")
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'extracted_text': extracted_text,
            'text_length': len(extracted_text),
            'blood_analysis': blood_analysis,
            'message': 'Blood report analyzed successfully'
        })
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR in upload:")
        traceback.print_exc()
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/api/query', methods=['POST'])
def query_report():
    """
    Answer questions about blood report (SIMPLE - NO LLM)
    """
    try:
        data = request.get_json()
        
        if 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query'].lower()
        file_id = data.get('file_id', '')
        
        # Get report if available
        if not file_id or file_id not in report_storage:
            return jsonify({
                'success': True,
                'response': "Please upload a blood report first! Click 'Upload Report' and select your report.",
                'sources': []
            })
        
        report_data = report_storage[file_id]
        report_text = report_data['text']
        blood_analysis = report_data.get('analysis', {})
        
        # Use RAG + LLM for intelligent responses
        print(f"\n💬 Query: {query}")
        
        # Step 1: Retrieve medical knowledge (offline RAG)
        print("🔍 Searching medical knowledge base...")
        rag_results = rag_retriever.retrieve(query, n_results=3)
        
        if rag_results.get('error'):
            print(f"⚠️ RAG not available: {rag_results['error']}")
            context = ""
            sources = []
        else:
            context = rag_results.get('context', '')
            sources = rag_results.get('sources', [])
            print(f"📚 Found {len(sources)} sources")
        
        # Step 2: Prepare blood values for LLM
        blood_values = {}
        if blood_analysis and blood_analysis.get('success'):
            for result in blood_analysis.get('results', []):
                blood_values[result['test_name']] = {
                    'value': result['value'],
                    'unit': result['unit'],
                    'status': result['interpretation']['status']
                }
        
        # Step 3: Generate response using RAG + LLM (or fallback to keywords)
        if local_llm.model is not None:
            print("🤖 Generating response with local LLM...")
            # Build context string with blood values
            blood_context = ""
            if blood_values:
                blood_context = "Patient's Blood Test Results:\n"
                for test_name, test_data in blood_values.items():
                    blood_context += f"- {test_name}: {test_data['value']} {test_data['unit']} ({test_data['status']})\n"
            
            full_context = f"{context}\n\n{blood_context}".strip() if context else blood_context
            response = local_llm.answer_question(
                question=query,
                context=full_context if full_context else None,
                max_length=250
            )
        else:
            print("⚡ Using keyword matching (LLM not loaded)...")
            response = generate_simple_response(query, blood_analysis, report_text)
        
        print("✅ Response generated")
        
        return jsonify({
            'success': True,
            'response': response,
            'sources': sources if sources else ['Your blood report analysis']
        })
    
    except Exception as e:
        import traceback
        print(f"\n❌ ERROR in query:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


def generate_simple_response(query: str, blood_analysis: dict, report_text: str) -> str:
    """
    Generate simple response based on keywords (NO LLM NEEDED)
    """
    
    # Summary request
    if any(word in query for word in ['summary', 'overview', 'what info', 'results', 'what did']):
        if not blood_analysis or not blood_analysis.get('success'):
            return f"📄 **Report Text Extracted:**\n\n{report_text[:500]}...\n\nThis appears to be a medical report. I can help answer questions about blood test values if this is a blood test report."
        
        response = "📋 **Blood Report Summary**\n\n"
        response += f"**Tests Analyzed:** {blood_analysis['total_tests']}\n"
        response += f"**Abnormal Results:** {blood_analysis['abnormal_count']}\n"
        response += f"**Overall Status:** {blood_analysis['overall_status'].replace('_', ' ').title()}\n\n"
        
        if blood_analysis['abnormal_count'] > 0:
            response += "**⚠️ Values Outside Normal Range:**\n"
            for result in blood_analysis['results']:
                if result['interpretation']['status'] != 'normal':
                    response += f"• {result['display_name']}: {result['value']} {result['unit']} "
                    response += f"[{result['interpretation']['status'].upper()}]\n"
        else:
            response += "✅ **All values within normal ranges!**\n"
        
        return response
    
    # Specific test queries
    test_keywords = {
        'wbc': ['wbc', 'white blood', 'leucocyte'],
        'hemoglobin': ['hemoglobin', 'hb ', 'hgb', 'haemoglobin'],
        'rbc': ['rbc', 'red blood'],
        'platelet': ['platelet', 'plt'],
        'glucose': ['glucose', 'sugar', 'blood sugar'],
        'cholesterol': ['cholesterol'],
        'ldl': ['ldl', 'bad cholesterol'],
        'hdl': ['hdl', 'good cholesterol'],
        'triglycerides': ['triglycerides']
    }
    
    if blood_analysis and blood_analysis.get('success'):
        for test_name, keywords in test_keywords.items():
            if any(kw in query for kw in keywords):
                for result in blood_analysis['results']:
                    if result['test_name'] == test_name:
                        return format_test_result(result)
    
    # Default response
    return ("I can help you understand your blood report!\n\n"
            "**Try asking:**\n"
            "• 'What is my WBC?'\n"
            "• 'Is my hemoglobin normal?'\n"
            "• 'Show me the summary'\n"
            "• 'What is my glucose level?'")


def format_test_result(result: dict) -> str:
    """Format a single test result"""
    interp = result['interpretation']
    
    response = f"🩸 **{result['display_name']}:** {result['value']} {result['unit']}\n\n"
    
    status_emoji = "✅" if interp['status'] == 'normal' else "⚠️"
    response += f"{status_emoji} **Status:** {interp['status'].upper()}\n"
    response += f"📊 **Normal Range:** {interp.get('normal_range', 'Not available')}\n\n"
    
    if interp['status'] != 'normal':
        # Get interpretation text if available
        interpretation_text = interp.get('interpretation', interp.get('message', 'Value is outside normal range'))
        response += "**What this means:**\n"
        response += f"{interpretation_text}\n\n"
        
        if result.get('recommendations'):
            response += "**Recommendations:**\n"
            for rec in result['recommendations'][:3]:
                response += f"• {rec}\n"
    else:
        response += "Your value is within the normal healthy range.\n"
    
    return response


if __name__ == '__main__':
    # Create directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    # Run the app
    print("\n🚀 Starting server on http://localhost:5000")
    print("📝 API Endpoints:")
    print("   GET  /api/health  - Health check")
    print("   POST /api/upload  - Upload blood report")
    print("   POST /api/query   - Ask questions\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
