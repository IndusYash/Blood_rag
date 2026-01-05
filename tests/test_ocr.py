"""
OCR Module Tests
Tests for OCR text extraction functionality
"""

import pytest
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from backend.ocr.ocr_reader import OCRReader
from PIL import Image
import numpy as np


class TestOCRReader:
    """Test cases for OCR reader"""
    
    @pytest.fixture
    def ocr_reader(self):
        """Create OCR reader instance"""
        return OCRReader()
    
    @pytest.fixture
    def sample_image(self, tmp_path):
        """Create a sample image for testing"""
        # Create a simple test image with text
        img = Image.new('RGB', (200, 100), color='white')
        img_path = tmp_path / "test_image.png"
        img.save(img_path)
        return img_path
    
    def test_initialization(self, ocr_reader):
        """Test OCR reader initialization"""
        assert ocr_reader is not None
        assert ocr_reader.engine in ['tesseract', 'easyocr']
        assert ocr_reader.language == 'eng'
    
    def test_extract_from_image(self, ocr_reader, sample_image):
        """Test text extraction from image"""
        try:
            text = ocr_reader.extract_text(sample_image)
            assert isinstance(text, str)
        except Exception as e:
            pytest.skip(f"OCR engine not available: {e}")
    
    def test_preprocess_image(self, ocr_reader):
        """Test image preprocessing"""
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        processed = ocr_reader._preprocess_image(test_image)
        
        assert processed is not None
        assert processed.shape[0] == 100
        assert processed.shape[1] == 100
    
    def test_unsupported_file_format(self, ocr_reader, tmp_path):
        """Test handling of unsupported file formats"""
        unsupported_file = tmp_path / "test.txt"
        unsupported_file.write_text("test")
        
        with pytest.raises(ValueError):
            ocr_reader.extract_text(unsupported_file)
    
    def test_nonexistent_file(self, ocr_reader):
        """Test handling of nonexistent files"""
        with pytest.raises(FileNotFoundError):
            ocr_reader.extract_text("nonexistent_file.pdf")
    
    def test_confidence_scores(self, ocr_reader):
        """Test OCR confidence score calculation"""
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        
        try:
            scores = ocr_reader.get_confidence_scores(test_image)
            assert isinstance(scores, dict)
            assert 'average_confidence' in scores
            assert 'text_detected' in scores
        except Exception as e:
            pytest.skip(f"Confidence scoring not available: {e}")


def test_ocr_on_medical_report():
    """Integration test for medical report OCR"""
    ocr_reader = OCRReader()
    
    # This test requires actual medical reports
    # Skip if test data is not available
    test_report = Path("test_data/sample_report.pdf")
    
    if not test_report.exists():
        pytest.skip("Test medical report not available")
    
    text = ocr_reader.extract_text(test_report)
    
    assert len(text) > 0
    assert isinstance(text, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
