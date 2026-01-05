"""
Embeddings module for Medical Report AI
Creates vector embeddings for RAG knowledge base
"""

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pathlib import Path
import PyPDF2
import json
from typing import List, Dict
from tqdm import tqdm

from config import Config


class EmbeddingsGenerator:
    """Generate and store embeddings for medical documents"""
    
    def __init__(self, model_name: str = None):
        """
        Initialize embeddings generator
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        
        # Initialize ChromaDB
        self.client = chromadb.Client(Settings(
            persist_directory=Config.CHROMA_PERSIST_DIRECTORY,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=Config.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or Config.CHUNK_SIZE
        overlap = overlap or Config.CHUNK_OVERLAP
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            if chunk.strip():
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def embed_documents(self, docs_path: Path = None) -> Dict:
        """
        Process all documents and create embeddings
        
        Args:
            docs_path: Path to documents directory
            
        Returns:
            Statistics about embedded documents
        """
        docs_path = docs_path or Config.RAG_DOCS_PATH
        
        if not docs_path.exists():
            raise FileNotFoundError(f"Documents path not found: {docs_path}")
        
        stats = {
            'total_files': 0,
            'total_chunks': 0,
            'files_processed': []
        }
        
        # Process all PDF files
        pdf_files = list(docs_path.glob('*.pdf'))
        
        for pdf_file in tqdm(pdf_files, desc="Embedding documents"):
            # Extract text
            text = self.extract_text_from_pdf(pdf_file)
            
            if not text.strip():
                continue
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Create embeddings
            embeddings = self.model.encode(chunks, show_progress_bar=False)
            
            # Store in ChromaDB
            ids = [f"{pdf_file.stem}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    'source': str(pdf_file.name),
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            stats['total_files'] += 1
            stats['total_chunks'] += len(chunks)
            stats['files_processed'].append(pdf_file.name)
        
        return stats
    
    def query(self, query_text: str, n_results: int = None) -> Dict:
        """
        Query the vector database
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            
        Returns:
            Query results with documents and metadata
        """
        n_results = n_results or Config.TOP_K_RESULTS
        
        # Create query embedding
        query_embedding = self.model.encode([query_text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'][0],
            'metadatas': results['metadatas'][0],
            'distances': results['distances'][0]
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection"""
        return {
            'collection_name': self.collection.name,
            'count': self.collection.count(),
            'model': self.model_name
        }


def main():
    """Main function to build embeddings"""
    print("Initializing embeddings generator...")
    generator = EmbeddingsGenerator()
    
    print("Embedding documents...")
    stats = generator.embed_documents()
    
    print("\n=== Embedding Statistics ===")
    print(f"Files processed: {stats['total_files']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Files: {', '.join(stats['files_processed'])}")
    
    print("\n=== Collection Stats ===")
    collection_stats = generator.get_collection_stats()
    print(f"Collection: {collection_stats['collection_name']}")
    print(f"Document count: {collection_stats['count']}")
    print(f"Model: {collection_stats['model']}")


if __name__ == "__main__":
    main()
