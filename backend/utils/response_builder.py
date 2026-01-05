"""
Response Builder module for Medical Report AI
Generates AI responses using LLM with RAG context
"""

import os
from typing import Dict, List, Optional
from datetime import datetime

from config import Config


class ResponseBuilder:
    """Build AI responses for medical queries"""
    
    def __init__(self, llm_provider: str = None, model: str = None):
        """
        Initialize response builder
        
        Args:
            llm_provider: LLM provider ('openai', 'anthropic', or 'local')
            model: Model name to use
        """
        self.llm_provider = llm_provider or self._detect_llm_provider()
        self.model = model or Config.LLM_MODEL
        self.temperature = Config.LLM_TEMPERATURE
        self.max_tokens = Config.MAX_TOKENS
        
        # Initialize LLM client
        self._init_llm_client()
    
    def _detect_llm_provider(self) -> str:
        """Detect which LLM provider to use based on available API keys"""
        if Config.OPENAI_API_KEY:
            return 'openai'
        elif Config.ANTHROPIC_API_KEY:
            return 'anthropic'
        else:
            return 'local'  # Fallback to local/mock
    
    def _init_llm_client(self):
        """Initialize LLM client based on provider"""
        self.client = None
        
        if self.llm_provider == 'openai' and Config.OPENAI_API_KEY:
            try:
                import openai
                self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            except ImportError:
                print("OpenAI package not installed")
        
        elif self.llm_provider == 'anthropic' and Config.ANTHROPIC_API_KEY:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=Config.ANTHROPIC_API_KEY)
            except ImportError:
                print("Anthropic package not installed")
    
    def generate_response(
        self,
        query: str,
        context: Dict,
        file_id: str = None,
        report_text: str = None
    ) -> str:
        """
        Generate AI response using LLM with RAG context
        
        Args:
            query: User query
            context: Retrieved context from RAG
            file_id: Optional file identifier
            report_text: Optional extracted report text
            
        Returns:
            Generated response
        """
        # Build prompt with context
        prompt = self._build_prompt(query, context, report_text)
        
        # Generate response based on provider
        if self.llm_provider == 'openai' and self.client:
            return self._generate_openai_response(prompt)
        elif self.llm_provider == 'anthropic' and self.client:
            return self._generate_anthropic_response(prompt)
        else:
            return self._generate_local_response(query, context)
    
    def _build_prompt(
        self,
        query: str,
        context: Dict,
        report_text: str = None
    ) -> str:
        """
        Build prompt for LLM
        
        Args:
            query: User query
            context: RAG context
            report_text: Optional report text
            
        Returns:
            Formatted prompt
        """
        prompt_parts = [
            "You are a medical AI assistant helping users understand their medical reports.",
            "Use the following context to answer the user's question accurately and helpfully.",
            "",
            "IMPORTANT GUIDELINES:",
            "- Provide clear, accurate medical information",
            "- Reference the context sources when applicable",
            "- If uncertain, suggest consulting a healthcare provider",
            "- Use simple language for better understanding",
            "- Consider Indian healthcare context when relevant",
            "",
        ]
        
        # Add context
        if context.get('context'):
            prompt_parts.append("CONTEXT FROM MEDICAL KNOWLEDGE BASE:")
            prompt_parts.append(context['context'])
            prompt_parts.append("")
        
        # Add report text if available
        if report_text:
            prompt_parts.append("USER'S MEDICAL REPORT:")
            prompt_parts.append(report_text[:1000])  # Limit to first 1000 chars
            prompt_parts.append("")
        
        # Add user query
        prompt_parts.append("USER QUESTION:")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("RESPONSE:")
        
        return "\n".join(prompt_parts)
    
    def _generate_openai_response(self, prompt: str) -> str:
        """Generate response using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful medical AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_response()
    
    def _generate_anthropic_response(self, prompt: str) -> str:
        """Generate response using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._generate_fallback_response()
    
    def _generate_local_response(self, query: str, context: Dict) -> str:
        """Generate simple response without external LLM"""
        # Simple rule-based response as fallback
        response_parts = [
            f"Based on your question: '{query}'",
            "",
            "I found the following relevant information:",
            ""
        ]
        
        if context.get('documents'):
            for i, doc in enumerate(context['documents'][:3], 1):
                response_parts.append(f"{i}. {doc[:200]}...")
                response_parts.append("")
        else:
            response_parts.append("No specific information found in the knowledge base.")
        
        response_parts.append("")
        response_parts.append("Note: For personalized medical advice, please consult with a healthcare professional.")
        
        return "\n".join(response_parts)
    
    def _generate_fallback_response(self) -> str:
        """Generate fallback response when LLM fails"""
        return (
            "I apologize, but I'm having trouble generating a detailed response at the moment. "
            "Please try again or consult with a healthcare professional for medical advice."
        )
    
    def summarize_report(self, report_text: str, report_type: str = None) -> Dict:
        """
        Generate summary of medical report
        
        Args:
            report_text: Extracted report text
            report_type: Type of report (optional)
            
        Returns:
            Dictionary with summary information
        """
        # Simple keyword extraction and summary
        keywords = self._extract_keywords(report_text)
        
        summary = {
            'report_type': report_type or 'Unknown',
            'text_length': len(report_text),
            'keywords': keywords,
            'generated_at': datetime.now().isoformat(),
            'summary': f"Report contains {len(report_text)} characters and discusses: {', '.join(keywords[:5])}"
        }
        
        return summary
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract medical keywords from text"""
        # Common medical terms
        medical_terms = [
            'normal', 'abnormal', 'elevated', 'low', 'high',
            'hemoglobin', 'glucose', 'cholesterol', 'pressure',
            'blood', 'test', 'result', 'level', 'count',
            'heart', 'lung', 'kidney', 'liver', 'thyroid'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in medical_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords
    
    def format_medical_values(self, values: Dict) -> str:
        """
        Format medical values for display
        
        Args:
            values: Dictionary of medical values
            
        Returns:
            Formatted string
        """
        formatted = []
        for key, value in values.items():
            formatted.append(f"{key}: {value}")
        
        return "\n".join(formatted)


def main():
    """Test response builder"""
    print("Response Builder Test")
    print("=" * 50)
    
    builder = ResponseBuilder()
    
    print(f"\nLLM Provider: {builder.llm_provider}")
    print(f"Model: {builder.model}")
    
    # Test query
    test_query = "What is a normal hemoglobin level?"
    test_context = {
        'context': 'Normal hemoglobin levels are 12-16 g/dL for females and 13-17 g/dL for males.',
        'sources': ['blood_test_info.pdf']
    }
    
    print(f"\nTest Query: {test_query}")
    response = builder.generate_response(test_query, test_context)
    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
