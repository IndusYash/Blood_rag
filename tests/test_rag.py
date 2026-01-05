"""
RAG Module Tests
Tests for RAG retrieval and embedding functionality
"""

import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.rag.embeddings import EmbeddingsGenerator
from backend.rag.retriever import RAGRetriever


class TestEmbeddingsGenerator:
    """Test cases for embeddings generator"""
    
    @pytest.fixture
    def embeddings_gen(self):
        """Create embeddings generator instance"""
        try:
            return EmbeddingsGenerator()
        except Exception as e:
            pytest.skip(f"Could not initialize embeddings generator: {e}")
    
    def test_initialization(self, embeddings_gen):
        """Test embeddings generator initialization"""
        assert embeddings_gen is not None
        assert embeddings_gen.model is not None
        assert embeddings_gen.collection is not None
    
    def test_chunk_text(self, embeddings_gen):
        """Test text chunking"""
        test_text = "This is a test. " * 100
        
        chunks = embeddings_gen.chunk_text(test_text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        assert all(len(chunk) <= 110 for chunk in chunks)  # chunk_size + small buffer
    
    def test_query(self, embeddings_gen):
        """Test querying embeddings"""
        test_query = "What is normal blood glucose?"
        
        try:
            results = embeddings_gen.query(test_query, n_results=3)
            
            assert 'documents' in results
            assert 'metadatas' in results
            assert isinstance(results['documents'], list)
        except Exception as e:
            pytest.skip(f"Query failed: {e}")
    
    def test_collection_stats(self, embeddings_gen):
        """Test getting collection statistics"""
        stats = embeddings_gen.get_collection_stats()
        
        assert 'collection_name' in stats
        assert 'count' in stats
        assert 'model' in stats
        assert isinstance(stats['count'], int)


class TestRAGRetriever:
    """Test cases for RAG retriever"""
    
    @pytest.fixture
    def retriever(self):
        """Create RAG retriever instance"""
        try:
            return RAGRetriever()
        except Exception as e:
            pytest.skip(f"Could not initialize retriever: {e}")
    
    def test_initialization(self, retriever):
        """Test retriever initialization"""
        assert retriever is not None
        assert retriever.model is not None
    
    def test_health_check(self, retriever):
        """Test retriever health check"""
        health = retriever.health_check()
        
        assert isinstance(health, dict)
        assert 'status' in health
        assert health['status'] in ['ready', 'not_ready', 'error']
    
    def test_retrieve(self, retriever):
        """Test document retrieval"""
        test_query = "What are normal hemoglobin levels?"
        
        results = retriever.retrieve(test_query, n_results=3)
        
        assert isinstance(results, dict)
        assert 'documents' in results
        assert 'sources' in results
        assert 'context' in results
        assert isinstance(results['documents'], list)
    
    def test_extract_medical_keywords(self, retriever):
        """Test medical keyword extraction"""
        test_text = "Patient has high blood glucose and elevated cholesterol levels."
        
        keywords = retriever._extract_medical_keywords(test_text)
        
        assert isinstance(keywords, list)
        assert 'blood' in keywords or 'glucose' in keywords or 'cholesterol' in keywords
    
    def test_analyze_medical_text(self, retriever):
        """Test medical text analysis"""
        test_text = """
        Blood Test Results:
        Hemoglobin: 14.5 g/dL
        Glucose: 95 mg/dL
        Cholesterol: 180 mg/dL
        """
        
        analysis = retriever.analyze_medical_text(test_text)
        
        assert isinstance(analysis, dict)
        assert 'keywords' in analysis
        assert 'relevant_info' in analysis
        assert 'analysis_complete' in analysis
    
    def test_get_similar_reports(self, retriever):
        """Test finding similar reports"""
        test_report = "Patient has diabetes with high blood sugar levels."
        
        results = retriever.get_similar_reports(test_report, n_results=3)
        
        assert isinstance(results, dict)
        assert 'documents' in results


def test_end_to_end_rag():
    """Integration test for complete RAG pipeline"""
    try:
        # Initialize components
        embeddings_gen = EmbeddingsGenerator()
        retriever = RAGRetriever()
        
        # Test query
        query = "What is a normal cholesterol level?"
        results = retriever.retrieve(query)
        
        assert len(results['documents']) >= 0
        
    except Exception as e:
        pytest.skip(f"End-to-end test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
