# Medical Report AI 🏥

An AI-powered medical report analysis system that uses OCR, computer vision, and RAG (Retrieval-Augmented Generation) to help understand medical reports.

## 🌟 Features

- **OCR Text Extraction**: Extract text from medical reports (PDF, images)
- **RAG-based Q&A**: Ask questions about your medical reports with context-aware answers
- **Report Classification**: Classify medical reports by type using Vision Transformer
- **Interactive Chat Interface**: User-friendly web interface for report analysis
- **Indian Healthcare Context**: Tailored for Indian health guidelines and standards

## 📁 Project Structure

```
MEDICAL_REPORT_AI/
├── backend/                 # Backend API + ML models
│   ├── app.py              # Flask API entry point
│   ├── config.py           # Configuration
│   ├── requirements.txt    # Python dependencies
│   ├── models/             # Trained models
│   ├── rag/                # RAG knowledge base
│   ├── ocr/                # OCR module
│   └── utils/              # Utility functions
├── frontend/               # React frontend
│   ├── src/                # Source code
│   ├── package.json        # Node dependencies
│   └── vite.config.js      # Vite configuration
├── training/               # Training scripts
├── tests/                  # Test files
├── dataset/                # Training datasets
└── docs/                   # Documentation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+
- Tesseract OCR (for text extraction)

### Backend Setup

1. Navigate to backend directory:
```bash
cd backend
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Initialize RAG knowledge base:
```bash
python rag/embeddings.py
```

6. Run the backend server:
```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Frontend Setup

1. Navigate to frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`

## 📖 Usage

### Upload a Medical Report

1. Open the web interface
2. Click "Upload Report"
3. Select a PDF or image file
4. Wait for OCR processing

### Ask Questions

1. After uploading, go to "Ask Questions" tab
2. Type your question about the report
3. Get AI-powered answers with medical context

### Example Questions

- "What are the key findings in my report?"
- "Are there any abnormal values?"
- "What does this cholesterol level mean?"
- "What should I discuss with my doctor?"

## 🧪 Testing

Run tests:
```bash
cd tests
pytest test_ocr.py -v
pytest test_rag.py -v
pytest test_api.py -v
```

## 🎓 Training Custom Models

### Train ViT Classifier

```bash
cd training
python train_vit.py
```

### Train YOLO Detector

```bash
python train_yolo.py
```

### Preprocess Data

```bash
python preprocess_data.py
```

## 📊 Model Performance

- **ViT Classifier**: Classifies report types (Blood Test, X-Ray, MRI, etc.)
- **YOLO Detector**: Detects tables and regions in reports (optional)
- **RAG System**: Retrieves relevant medical knowledge for Q&A

## 🔒 Privacy & Security

- All data processing happens locally
- No medical reports are stored permanently
- HIPAA compliance considerations included
- Secure file handling and validation

## ⚕️ Medical Disclaimer

**This is an educational project and should NOT be used as a substitute for professional medical advice.**

Always consult with qualified healthcare professionals for:
- Medical diagnosis
- Treatment decisions
- Health concerns
- Report interpretation

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is for educational purposes. See LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work

## 🙏 Acknowledgments

- Medical knowledge base sources
- Indian Health Guidelines
- Open-source libraries and frameworks

## 📧 Contact

For questions or support:
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 🔗 Related Resources

- [National Health Portal India](https://www.nhp.gov.in/)
- [Ayushman Bharat](https://pmjay.gov.in/)
- [Medical Terminology Reference](https://medlineplus.gov/)

---

**Built with ❤️ for better healthcare accessibility**
