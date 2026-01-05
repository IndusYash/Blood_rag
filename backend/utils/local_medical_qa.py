"""
Local Medical Q&A using HuggingFace models (100% local, no API needed)
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional


class LocalMedicalQA:
    """
    Local medical Q&A using HuggingFace transformers
    No API keys needed - runs entirely on your machine
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small", use_model: bool = True):
        """
        Initialize local QA model
        
        Args:
            model_name: HuggingFace model to use
                - "google/flan-t5-small" (77M params, ~300MB, good for CPU)
                - "t5-small" (60M params, ~242MB, fastest)
                - None or use_model=False for rule-based only
            use_model: Whether to load the model (set False to skip for pure rule-based)
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
        if not use_model:
            print("🚀 Running in RULE-BASED mode (no model loading)")
            print("✅ Fast startup, instant responses!\n")
            return
        
        print(f"\n🔄 Loading {model_name} from HuggingFace...")
        print("⏳ This may take a few moments on first run (downloading ~300MB)...")
        print("💡 To skip model loading, set use_model=False in config\n")
        
        self.device = "cpu"  # Force CPU for lightweight systems
        print(f"💻 Running on: {self.device}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True      # Optimize for low memory
            )
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Memory usage: ~{self._get_model_size_mb():.1f} MB\n")
            
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")
            print("✅ Falling back to rule-based responses (works great!)\n")
            self.model = None
    
    def _get_model_size_mb(self) -> float:
        """Estimate model size in MB"""
        if self.model is None:
            return 0
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        return param_size / (1024 * 1024)
    
    def answer_question(
        self,
        question: str,
        context: str = None,
        max_length: int = 200
    ) -> str:
        """
        Answer medical question using local model
        
        Args:
            question: User's question
            context: Additional context (optional)
            max_length: Maximum response length
            
        Returns:
            Generated answer
        """
        if self.model is None:
            return "Model not loaded. Running in rule-based mode."
        
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Answer this medical question: {question}"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(self.device)
            
            # Generate response with CPU-optimized settings
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=20,
                    do_sample=False,        # Faster on CPU (no sampling)
                    num_beams=2,            # Reduced beams for speed
                    early_stopping=True,
                    no_repeat_ngram_size=3  # Avoid repetition
                )
            
            # Decode response
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return answer.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error processing your question. Please try again."
    
    def analyze_blood_report_with_context(
        self,
        question: str,
        blood_data: Dict,
        max_length: int = 300
    ) -> str:
        """
        Answer questions about blood report using structured data
        
        Args:
            question: User's question
            blood_data: Structured blood test results
            max_length: Maximum response length
            
        Returns:
            Generated answer with blood data context
        """
        # Build context from blood data
        context_parts = []
        
        if blood_data.get('total_tests'):
            context_parts.append(f"Total tests: {blood_data['total_tests']}")
        
        if blood_data.get('abnormal_count'):
            context_parts.append(f"Abnormal values: {blood_data['abnormal_count']}")
        
        if blood_data.get('results'):
            context_parts.append("Test results:")
            for result in blood_data['results'][:10]:  # Limit to 10 tests
                status = result.get('interpretation', {}).get('status', 'unknown')
                context_parts.append(
                    f"- {result['display_name']}: {result['value']} {result['unit']} [{status}]"
                )
        
        context = "\n".join(context_parts)
        
        return self.answer_question(question, context, max_length)


class MedicalResponseGenerator:
    """
    Main interface for generating medical responses
    Combines local model + rule-based analysis
    """
    
    def __init__(self, use_model: bool = False):
        """
        Initialize with optional local model
        
        Args:
            use_model: Set to True to load HuggingFace model (slow startup, better responses)
                      Set to False for instant startup with rule-based responses (recommended for CPU)
        """
        if use_model:
            # Try to load FLAN-T5-small (fast, good quality)
            self.qa_model = LocalMedicalQA("google/flan-t5-small", use_model=True)
            self.is_model_loaded = self.qa_model.model is not None
        else:
            # Rule-based only (instant startup)
            print("🚀 Medical Response Generator initialized in RULE-BASED mode")
            print("✅ Fast responses, no model loading needed!\n")
            self.qa_model = LocalMedicalQA(use_model=False)
            self.is_model_loaded = False
    
    def generate_response(
        self,
        query: str,
        report_text: str = None,
        blood_analysis: Dict = None,
        rag_context: Dict = None
    ) -> str:
        """
        Generate intelligent response using local model or rules
        
        Args:
            query: User's question
            report_text: Extracted report text
            blood_analysis: Structured blood analysis
            rag_context: Retrieved medical knowledge
            
        Returns:
            Generated response
        """
        if not self.is_model_loaded:
            return self._fallback_response(query, blood_analysis)
        
        # If we have blood analysis, use it with the model
        if blood_analysis and blood_analysis.get('success'):
            return self.qa_model.analyze_blood_report_with_context(
                query,
                blood_analysis
            )
        
        # Build context from RAG if available
        context = ""
        if rag_context and rag_context.get('documents'):
            context = "\n".join(rag_context['documents'][:2])
        
        # General medical question
        return self.qa_model.answer_question(query, context)
    
    def _fallback_response(self, query: str, blood_analysis: Dict = None) -> str:
        """Fallback rule-based response (works great without model!)"""
        query_lower = query.lower()
        
        # Blood report summary
        if blood_analysis and blood_analysis.get('success'):
            response = f"📊 **Blood Report Analysis:**\n\n"
            response += f"Tests Found: {blood_analysis['total_tests']}\n"
            response += f"Abnormal Values: {blood_analysis['abnormal_count']}\n\n"
            
            if blood_analysis['abnormal_count'] > 0:
                response += "⚠️ **Abnormal Results:**\n"
                for result in blood_analysis['results'][:5]:
                    if result['interpretation']['status'] != 'normal':
                        response += f"• {result['display_name']}: {result['value']} {result['unit']}"
                        response += f" [{result['interpretation']['status'].upper()}]\n"
            else:
                response += "✅ All values are within normal ranges!\n"
            
            return response
        
        # General medical questions - provide medical knowledge
        if 'hemoglobin' in query_lower:
            return ("Hemoglobin is a protein in red blood cells that carries oxygen throughout your body. "
                   "Normal levels are 13-17 g/dL for men and 12-15 g/dL for women. "
                   "Low hemoglobin (anemia) causes fatigue and weakness. "
                   "High hemoglobin may indicate dehydration or lung disease.")
        
        if 'wbc' in query_lower or 'white blood' in query_lower:
            return ("White Blood Cells (WBC) are part of your immune system that fight infections. "
                   "Normal range: 4,000-11,000/µL. "
                   "High WBC may indicate infection or inflammation. "
                   "Low WBC may mean weakened immune system.")
        
        if 'cholesterol' in query_lower:
            return ("Cholesterol is a fat-like substance in your blood. "
                   "Total cholesterol should be <200 mg/dL. "
                   "LDL (bad) should be <100 mg/dL. HDL (good) should be >40 mg/dL. "
                   "High cholesterol increases heart disease risk. "
                   "Manage with diet, exercise, and medication if needed.")
        
        if 'glucose' in query_lower or 'sugar' in query_lower or 'diabetes' in query_lower:
            return ("Blood glucose (sugar) provides energy to your cells. "
                   "Fasting glucose: 70-100 mg/dL (normal), 100-125 (pre-diabetes), >126 (diabetes). "
                   "HbA1c: <5.7% (normal), 5.7-6.4% (pre-diabetes), ≥6.5% (diabetes). "
                   "Control blood sugar with diet, exercise, and medication if prescribed.")
        
        # Default
        return ("I can help you understand your blood test results! "
               "Upload a blood report and ask me:\n"
               "• 'What is my WBC count?'\n"
               "• 'Is my hemoglobin normal?'\n"
               "• 'What does high cholesterol mean?'\n"
               "• 'Summarize my report'")


# Quick test function
def test_local_model():
    """Test the local model"""
    print("="*60)
    print("Testing Local Medical QA - Rule-Based Mode")
    print("="*60)
    
    qa = MedicalResponseGenerator(use_model=False)
    
    # Test questions
    questions = [
        "What is hemoglobin?",
        "Why is high WBC concerning?",
        "What are normal cholesterol levels?"
    ]
    
    for q in questions:
        print(f"\n❓ Q: {q}")
        answer = qa.generate_response(q)
        print(f"💡 A: {answer[:150]}...")
    
    print("\n" + "="*60)
    print("✅ Test complete!")


if __name__ == "__main__":
    test_local_model()
