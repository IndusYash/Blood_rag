"""
Medical Report AI - Backend API
Flask/FastAPI entry point for processing medical reports
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from pathlib import Path

from config import Config
from ocr.ocr_reader import OCRReader
from rag.retriever import RAGRetriever
from utils.file_handler import FileHandler
from utils.response_builder import ResponseBuilder
from utils.blood_report_analyzer import BloodReportAnalyzer
from utils.local_medical_qa import MedicalResponseGenerator

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(Config)

# Initialize components
print("🚀 Initializing Medical Report AI...")
ocr_reader = OCRReader()
rag_retriever = RAGRetriever()
file_handler = FileHandler()
response_builder = ResponseBuilder()
blood_analyzer = BloodReportAnalyzer()

# Initialize LOCAL AI with FLAN-T5 model for intelligent analysis
print("\n" + "="*60)
print("🏥 Medical Report AI - Starting...")
print("="*60)
print("🤖 Loading AI model for medical analysis...")
# FLAN-T5-small: 77M params, ~300MB, optimized for CPU
medical_qa = MedicalResponseGenerator(use_model=True)
print("="*60)
print("✅ System Ready!\n")

# Store extracted texts in memory (in production, use database)
report_storage = {}


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'Medical Report AI API is running'
    })


@app.route('/api/upload', methods=['POST'])
def upload_report():
    """
    Upload and process a medical report
    Accepts: PDF, PNG, JPG, JPEG files
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        file_path = file_handler.save_uploaded_file(file)
        print(f"📁 File saved to: {file_path}")
        
        # Extract text using OCR
        print(f"🔍 Starting OCR extraction...")
        extracted_text = ocr_reader.extract_text(file_path)
        print(f"📝 OCR extracted {len(extracted_text)} characters")
        if len(extracted_text) == 0:
            print("⚠️ WARNING: OCR extracted 0 characters - check Tesseract installation!")
            print(f"   File type: {Path(file_path).suffix}")
            print(f"   File exists: {Path(file_path).exists()}")
        
        # Generate file ID
        file_id = Path(file_path).stem
        
        # Store extracted text with file_id for later queries
        report_storage[file_id] = {
            'text': extracted_text,
            'file_path': file_path
        }
        
        # Analyze blood report automatically
        blood_analysis = blood_analyzer.analyze_report(extracted_text, gender='normal')
        
        return jsonify({
            'success': True,
            'file_id': file_id,
            'extracted_text': extracted_text,
            'blood_analysis': blood_analysis if blood_analysis.get('success') else None,
            'message': 'Report processed successfully'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query', methods=['POST'])
def query_report():
    """
    Query the uploaded report with intelligent local analysis
    NO LLM API KEY NEEDED - Uses rule-based analyzer
    """
    try:
        data = request.get_json()
        
        if 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        file_id = data.get('file_id', '')
        
        # Get the uploaded report text if available
        report_text = ''
        blood_analysis = None
        
        if file_id and file_id in report_storage:
            report_text = report_storage[file_id]['text']
            # Get or create blood analysis
            if 'analysis' not in report_storage[file_id]:
                blood_analysis = blood_analyzer.analyze_report(report_text, gender='normal')
                if blood_analysis.get('success'):
                    report_storage[file_id]['analysis'] = blood_analysis
            else:
                blood_analysis = report_storage[file_id]['analysis']
        
        # If no report uploaded yet
        if not report_text:
            return jsonify({
                'success': True,
                'response': "Please upload a blood report first! Click on 'Upload Report' tab and select your report file.",
                'sources': []
            })
        
        query_lower = query.lower()
        
        # ========== HANDLE SPECIFIC VALUE QUERIES ==========
        
        # WBC/White Blood Cell
        if any(term in query_lower for term in ['wbc', 'white blood', 'leucocyte']):
            if blood_analysis and blood_analysis.get('success'):
                for result in blood_analysis['results']:
                    if result['test_name'] == 'wbc':
                        interp = result['interpretation']
                        response = f"🩸 **Your WBC Count:** {result['value']} {result['unit']}\n\n"
                        
                        if interp['status'] == 'normal':
                            response += f"✅ **Status:** NORMAL\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "Your white blood cell count is healthy. WBCs help fight infections."
                        elif interp['status'] == 'low':
                            response += f"⚠️ **Status:** LOW\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "**What this means:**\n"
                            response += "• Low WBC count (leukopenia) means weakened immune system\n"
                            response += "• May be caused by viral infections, bone marrow issues, or certain medications\n\n"
                            response += "**Recommendations:**\n"
                            for rec in result['recommendations'][:3]:
                                response += f"• {rec}\n"
                        else:  # high
                            response += f"⚠️ **Status:** HIGH\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "**What this means:**\n"
                            response += "• Elevated WBC count (leukocytosis) often indicates:\n"
                            response += "  - Active infection or inflammation\n"
                            response += "  - Immune response to stress\n"
                            response += "  - Possible allergic reaction\n\n"
                            response += "**Recommendations:**\n"
                            for rec in result['recommendations'][:3]:
                                response += f"• {rec}\n"
                        
                        return jsonify({
                            'success': True,
                            'response': response,
                            'sources': ['Your blood report analysis']
                        })
            
            return jsonify({
                'success': True,
                'response': "I found WBC mentioned in your report but couldn't extract the exact value. The text might not be clear enough. Try uploading a clearer image.",
                'sources': ['Your uploaded report']
            })
        
        # Hemoglobin
        if any(term in query_lower for term in ['hemoglobin', 'hb ', 'hgb']):
            if blood_analysis and blood_analysis.get('success'):
                for result in blood_analysis['results']:
                    if result['test_name'] == 'hemoglobin':
                        interp = result['interpretation']
                        response = f"🩸 **Your Hemoglobin:** {result['value']} {result['unit']}\n\n"
                        
                        if interp['status'] == 'normal':
                            response += f"✅ **Status:** NORMAL\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "Your hemoglobin level is healthy. Hemoglobin carries oxygen throughout your body."
                        elif interp['status'] == 'low':
                            response += f"⚠️ **Status:** LOW (Anemia)\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "**What this means:**\n"
                            response += "• You have anemia - not enough red blood cells or hemoglobin\n"
                            response += "• May cause: fatigue, weakness, pale skin, shortness of breath\n\n"
                            response += "**Recommendations:**\n"
                            for rec in result['recommendations']:
                                response += f"• {rec}\n"
                        else:  # high
                            response += f"⚠️ **Status:** HIGH\n"
                            response += f"📊 **Normal Range:** {interp['normal_range']}\n\n"
                            response += "**What this means:**\n"
                            response += "• Elevated hemoglobin (polycythemia)\n"
                            response += "• May be due to dehydration or living at high altitude\n\n"
                            response += "**Recommendations:**\n"
                            for rec in result['recommendations']:
                                response += f"• {rec}\n"
                        
                        return jsonify({
                            'success': True,
                            'response': response,
                            'sources': ['Your blood report analysis']
                        })
            
            return jsonify({
                'success': True,
                'response': "I couldn't find a clear hemoglobin value in your report. Make sure the report includes CBC (Complete Blood Count) results.",
                'sources': ['Your uploaded report']
            })
        
        # Cholesterol
        if any(term in query_lower for term in ['cholesterol', 'ldl', 'hdl', 'lipid']):
            if blood_analysis and blood_analysis.get('success'):
                cholesterol_results = []
                for result in blood_analysis['results']:
                    if result['test_name'] in ['cholesterol', 'ldl', 'hdl', 'triglycerides']:
                        cholesterol_results.append(result)
                
                if cholesterol_results:
                    response = "💊 **Your Lipid Profile:**\n\n"
                    for result in cholesterol_results:
                        interp = result['interpretation']
                        status_emoji = "✅" if interp['status'] == 'normal' else "⚠️"
                        response += f"{status_emoji} **{result['display_name']}:** {result['value']} {result['unit']}"
                        response += f" [{interp['status'].upper()}]\n"
                    
                    response += "\n**Understanding Your Results:**\n"
                    response += "• **Total Cholesterol:** Should be <200 mg/dL\n"
                    response += "• **LDL (Bad):** Should be <100 mg/dL\n"
                    response += "• **HDL (Good):** Should be >40 mg/dL (men) or >50 mg/dL (women)\n"
                    response += "• **Triglycerides:** Should be <150 mg/dL\n\n"
                    
                    has_high = any(r['interpretation']['status'] == 'high' for r in cholesterol_results)
                    if has_high:
                        response += "**💡 Recommendations for High Cholesterol:**\n"
                        response += "• Reduce saturated fats (butter, red meat, fried foods)\n"
                        response += "• Increase fiber (oats, beans, vegetables)\n"
                        response += "• Exercise 30 minutes daily\n"
                        response += "• Consult doctor about statins if needed\n"
                    
                    return jsonify({
                        'success': True,
                        'response': response,
                        'sources': ['Your blood report analysis']
                    })
        
        # Glucose/Diabetes
        if any(term in query_lower for term in ['glucose', 'sugar', 'diabetes', 'diabetic', 'hba1c']):
            if blood_analysis and blood_analysis.get('success'):
                glucose_results = []
                for result in blood_analysis['results']:
                    if result['test_name'] in ['fasting_glucose', 'random_glucose', 'hba1c']:
                        glucose_results.append(result)
                
                if glucose_results:
                    response = "🍬 **Your Blood Glucose Levels:**\n\n"
                    for result in glucose_results:
                        interp = result['interpretation']
                        status_emoji = "✅" if interp['status'] == 'normal' else "⚠️"
                        response += f"{status_emoji} **{result['display_name']}:** {result['value']} {result['unit']}"
                        response += f" [{interp['status'].upper()}]\n"
                    
                    response += "\n**Understanding Your Results:**\n"
                    response += "• **Fasting Glucose:** 70-100 mg/dL (Normal), 100-125 (Pre-diabetes), >126 (Diabetes)\n"
                    response += "• **HbA1c:** <5.7% (Normal), 5.7-6.4% (Pre-diabetes), ≥6.5% (Diabetes)\n\n"
                    
                    has_high = any(r['interpretation']['status'] == 'high' for r in glucose_results)
                    if has_high:
                        response += "**💡 Recommendations for High Blood Sugar:**\n"
                        response += "• Reduce refined carbs and sugar\n"
                        response += "• Increase physical activity\n"
                        response += "• Monitor blood glucose regularly\n"
                        response += "• Consult doctor for diabetes screening\n"
                        response += "• Consider medication if pre-diabetic/diabetic\n"
                    
                    return jsonify({
                        'success': True,
                        'response': response,
                        'sources': ['Your blood report analysis']
                    })
        
        # ========== GENERAL QUERIES ==========
        
        # "What info did you get" / "Summary"
        if any(phrase in query_lower for phrase in ['what info', 'what did', 'summary', 'overview', 'results']):
            if blood_analysis and blood_analysis.get('success'):
                response = "📋 **Blood Report Summary**\n\n"
                response += f"**Tests Analyzed:** {blood_analysis['total_tests']}\n"
                response += f"**Abnormal Results:** {blood_analysis['abnormal_count']}\n"
                response += f"**Overall Status:** {blood_analysis['overall_status'].replace('_', ' ').title()}\n\n"
                
                if blood_analysis['abnormal_count'] > 0:
                    response += "**⚠️ Values Outside Normal Range:**\n"
                    for result in blood_analysis['results']:
                        if result['interpretation']['status'] != 'normal':
                            response += f"• {result['display_name']}: {result['value']} {result['unit']}"
                            response += f" [{result['interpretation']['status'].upper()}]\n"
                    
                    response += "\n**You can ask me:**\n"
                    response += "• \"What is my WBC count?\"\n"
                    response += "• \"Why is my hemoglobin low?\"\n"
                    response += "• \"Is my cholesterol normal?\"\n"
                    response += "• \"What should I do about high glucose?\"\n"
                else:
                    response += "✅ **All your test values are within normal ranges!**\n\n"
                    response += "Keep maintaining a healthy lifestyle:\n"
                    response += "• Balanced diet\n"
                    response += "• Regular exercise\n"
                    response += "• Adequate sleep\n"
                    response += "• Regular check-ups\n"
                
                return jsonify({
                    'success': True,
                    'response': response,
                    'sources': ['Your blood report analysis']
                })
            else:
                return jsonify({
                    'success': True,
                    'response': f"I extracted this text from your report:\n\n{report_text[:500]}...\n\nIt doesn't appear to be a standard blood test report. Please upload a CBC, Lipid Profile, or Blood Glucose report.",
                    'sources': ['Your uploaded report']
                })
        
        # Fallback for other questions - USE LOCAL AI MODEL
        # Get RAG context for medical knowledge
        context = rag_retriever.retrieve(query)
        
        # Use local HuggingFace model to generate intelligent response
        if medical_qa.is_model_loaded:
            print(f"🤖 Using local AI model to answer: {query}")
            response = medical_qa.generate_response(
                query=query,
                report_text=report_text,
                blood_analysis=blood_analysis,
                rag_context=context
            )
            
            # Add tip if they have a report
            if report_text and not blood_analysis:
                response += "\n\n💡 I also have your report. Ask me:\n"
                response += "• \"What is my WBC?\"\n"
                response += "• \"Summarize my report\"\n"
            
            return jsonify({
                'success': True,
                'response': response,
                'sources': ['Local AI Model', *context.get('sources', [])]
            })
        
        # Fallback if model not loaded
        response = f"Based on medical knowledge:\n\n"
        if context.get('documents'):
            response += context['documents'][0]
        else:
            response += "I need more context to answer that question."
        
        if report_text:
            response += "\n\n💡 Ask me about YOUR report:\n"
            response += "• \"What is my WBC?\"\n"
            response += "• \"Is my hemoglobin normal?\"\n"
        
        return jsonify({
            'success': True,
            'response': response,
            'sources': context.get('sources', [])
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze', methods=['POST'])
def analyze_report():
    """
    Analyze medical report and provide insights
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        gender = data.get('gender', 'normal')  # Optional: male, female, or normal
        
        # Try blood report analysis first
        blood_analysis = blood_analyzer.analyze_report(report_text, gender)
        
        if blood_analysis.get('success'):
            # Generate readable report
            report_summary = blood_analyzer.generate_report_text(blood_analysis)
            
            return jsonify({
                'success': True,
                'analysis_type': 'blood_report',
                'analysis': blood_analysis,
                'summary': report_summary
            })
        
        # Fallback to general RAG analysis
        analysis = rag_retriever.analyze_medical_text(report_text)
        
        return jsonify({
            'success': True,
            'analysis_type': 'general',
            'analysis': analysis
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze-blood', methods=['POST'])
def analyze_blood_report():
    """
    Specifically analyze blood test reports with value interpretation
    """
    try:
        data = request.get_json()
        
        if 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        report_text = data['text']
        gender = data.get('gender', 'normal')
        
        # Analyze blood report
        analysis = blood_analyzer.analyze_report(report_text, gender)
        
        if not analysis.get('success'):
            return jsonify(analysis), 400
        
        # Generate readable summary
        summary = blood_analyzer.generate_report_text(analysis)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'summary': summary
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(Config.MODEL_PATH, exist_ok=True)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
