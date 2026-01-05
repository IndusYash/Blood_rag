"""
Retriever module for Medical Report AI
Query retrieval logic using RAG
"""

from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from config import Config


class RAGRetriever:
    """Retrieve relevant information from medical knowledge base"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize RAG retriever
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False
        ))
        
        # Get collection
        try:
            self.collection = self.client.get_collection(
                name=Config.CHROMA_COLLECTION_NAME
            )
        except Exception:
            # Collection doesn't exist yet
            self.collection = None
    
    def retrieve(self, query: str, n_results: int = None, filter_dict: Dict = None) -> Dict:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_dict: Metadata filters
            
        Returns:
            Dictionary containing retrieved documents and metadata
        """
        if self.collection is None:
            return {
                'documents': [],
                'sources': [],
                'context': '',
                'error': 'Knowledge base not initialized. Run embeddings.py first.'
            }
        
        n_results = n_results or Config.TOP_K_RESULTS
        
        # Create query embedding
        query_embedding = self.model.encode([query])[0].tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        # Format results
        documents = results['documents'][0] if results['documents'] else []
        metadatas = results['metadatas'][0] if results['metadatas'] else []
        distances = results['distances'][0] if results['distances'] else []
        
        # Extract unique sources
        sources = list(set([meta.get('source', 'Unknown') for meta in metadatas]))
        
        # Create context string
        context = self._format_context(documents, metadatas)
        
        return {
            'documents': documents,
            'metadatas': metadatas,
            'distances': distances,
            'sources': sources,
            'context': context,
            'query': query
        }
    
    def _format_context(self, documents: List[str], metadatas: List[Dict]) -> str:
        """
        Format retrieved documents into context string
        
        Args:
            documents: Retrieved document chunks
            metadatas: Metadata for each chunk
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        context_parts = []
        for i, (doc, meta) in enumerate(zip(documents, metadatas), 1):
            source = meta.get('source', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{doc}\n")
        
        return "\n".join(context_parts)
    
    def analyze_medical_text(self, text: str) -> Dict:
        """
        Analyze medical report text and provide insights
        
        Args:
            text: Medical report text
            
        Returns:
            Analysis results with relevant medical information
        """
        # Extract key terms (simple keyword extraction)
        keywords = self._extract_medical_keywords(text)
        
        # Retrieve relevant information for each keyword
        all_contexts = []
        for keyword in keywords[:5]:  # Limit to top 5 keywords
            results = self.retrieve(keyword, n_results=2)
            if results['documents']:
                all_contexts.extend(results['documents'])
        
        # Remove duplicates
        unique_contexts = list(set(all_contexts))
        
        return {
            'keywords': keywords,
            'relevant_info': unique_contexts[:5],  # Top 5 relevant pieces
            'text_length': len(text),
            'analysis_complete': True
        }
    
    def _extract_medical_keywords(self, text: str) -> List[str]:
        """
        Extract medical keywords from text
        
        Args:
            text: Input text
            
        Returns:
            List of medical keywords
        """
        # Common medical terms to look for
        medical_terms = [
            'blood', 'hemoglobin', 'glucose', 'cholesterol', 'pressure',
            'heart', 'lung', 'kidney', 'liver', 'diabetes', 'hypertension',
            'infection', 'inflammation', 'test', 'normal', 'abnormal',
            'ct', 'mri', 'xray', 'scan', 'ultrasound', 'ecg', 'ekg'
        ]
        
        text_lower = text.lower()
        found_keywords = []
        
        for term in medical_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords
    
    def get_similar_reports(self, report_text: str, n_results: int = 3) -> Dict:
        """
        Find similar medical reports
        
        Args:
            report_text: Medical report text
            n_results: Number of similar reports to find
            
        Returns:
            Similar reports and their metadata
        """
        return self.retrieve(report_text, n_results=n_results)
    
    def health_check(self) -> Dict:
        """Check if retriever is properly initialized"""
        if self.collection is None:
            return {
                'status': 'not_ready',
                'message': 'Collection not initialized'
            }
        
        try:
            count = self.collection.count()
            return {
                'status': 'ready',
                'document_count': count,
                'model': self.model_name,
                'collection_name': Config.CHROMA_COLLECTION_NAME
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }


def main():
    """Test the retriever"""
    print("Initializing RAG retriever...")
    retriever = RAGRetriever()
    
    # Health check
    health = retriever.health_check()
    print(f"\nHealth Check: {health}")
    
    if health['status'] == 'ready':
        # Test query
        test_query = "What is normal blood glucose level?"
        print(f"\nTest Query: {test_query}")
        
        results = retriever.retrieve(test_query)
        print(f"\nFound {len(results['documents'])} relevant documents")
        print(f"Sources: {', '.join(results['sources'])}")
        
        if results['documents']:
            print(f"\nTop Result:\n{results['documents'][0][:200]}...")


if __name__ == "__main__":
    main()
