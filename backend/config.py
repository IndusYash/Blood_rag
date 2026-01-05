"""
Configuration file for Medical Report AI
Contains paths, API keys, and database configurations
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent


class Config:
    """Application configuration"""
    
    # Flask config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
    
    # File upload settings
    UPLOAD_FOLDER = BASE_DIR / 'backend' / 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}
    
    # Model paths
    MODEL_PATH = BASE_DIR / 'backend' / 'models'
    YOLO_MODEL_PATH = MODEL_PATH / 'yolo_detector' / 'best.pt'
    VIT_MODEL_PATH = MODEL_PATH / 'vit_classifier' / 'model.pt'
    
    # RAG settings
    RAG_DOCS_PATH = BASE_DIR / 'backend' / 'rag' / 'docs'
    VECTOR_STORE_PATH = BASE_DIR / 'backend' / 'rag' / 'vector_store'
    EMBEDDING_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    TOP_K_RESULTS = 5
    
    # LLM API settings (for response generation)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')
    LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
    LLM_TEMPERATURE = 0.7
    MAX_TOKENS = 1000
    
    # OCR settings
    OCR_ENGINE = 'tesseract'  # Using Tesseract for fast, reliable OCR
    OCR_LANGUAGE = 'eng'
    OCR_CONFIG = '--oem 3 --psm 3'  # PSM 3: Fully automatic page segmentation
    
    # Database settings (optional - for storing reports)
    DATABASE_URI = os.getenv('DATABASE_URI', 'sqlite:///medical_reports.db')
    
    # ChromaDB settings
    CHROMA_COLLECTION_NAME = 'medical_knowledge'
    CHROMA_PERSIST_DIRECTORY = str(VECTOR_STORE_PATH)
    
    # Training settings
    DATASET_PATH = BASE_DIR / 'dataset'
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 50
    
    # Indian health guidelines specific
    INDIAN_UNITS = True  # Convert to Indian standard units if needed
    
    @staticmethod
    def init_app(app):
        """Initialize application with config"""
        pass


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Add production-specific settings


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
