"""
OCR Reader module for Medical Report AI
Extracts text from medical reports (PDF, images)
"""

import pytesseract
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from pdf2image import convert_from_path
import os
from typing import Union, List

from config import Config

# Configure Tesseract path at module level (Windows)
if os.name == 'nt':  # Windows
    tesseract_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Tesseract-OCR\tesseract.exe'
    ]
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            print(f"✅ Tesseract configured at: {path}")
            break
    else:
        print("⚠️ WARNING: Tesseract not found at standard locations!")
        print("   Install from: https://github.com/UB-Mannheim/tesseract/wiki")


class OCRReader:
    """Extract text from medical reports using OCR"""
    
    def __init__(self, engine: str = None):
        """
        Initialize OCR reader
        
        Args:
            engine: OCR engine to use ('tesseract', 'easyocr')
        """
        self.engine = engine or Config.OCR_ENGINE
        self.language = Config.OCR_LANGUAGE
        self.config = Config.OCR_CONFIG
        
        # Verify Tesseract is accessible
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract version: {version}")
            print(f"   Path: {pytesseract.pytesseract.tesseract_cmd}")
        except Exception as e:
            print(f"❌ Tesseract verification failed: {e}")
            print(f"   Current tesseract_cmd: {pytesseract.pytesseract.tesseract_cmd}")
        
        # Initialize EasyOCR if specified
        self.easyocr_reader = None
        if self.engine == 'easyocr':
            try:
                import easyocr
                self.easyocr_reader = easyocr.Reader(['en'])
            except ImportError:
                print("EasyOCR not installed. Falling back to Tesseract.")
                self.engine = 'tesseract'
    
    def extract_text(self, file_path: Union[str, Path]) -> str:
        """
        Extract text from file (PDF or image)
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text
        """
        file_path = Path(file_path)
        
        print(f"\n{'='*60}")
        print(f"🔍 OCR EXTRACTION STARTED")
        print(f"   File: {file_path.name}")
        print(f"   Engine: {self.engine}")
        print(f"{'='*60}")
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        ext = file_path.suffix.lower()
        print(f"   File type: {ext}")
        
        if ext == '.pdf':
            return self._extract_from_pdf(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            return self._extract_from_image(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _extract_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from PDF file
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        try:
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=300)
            
            # Extract text from each page
            all_text = []
            for i, image in enumerate(images):
                # Preprocess image
                processed_image = self._preprocess_image(np.array(image))
                
                # Extract text
                text = self._run_ocr(processed_image)
                all_text.append(f"--- Page {i+1} ---\n{text}\n")
            
            return "\n".join(all_text)
        
        except Exception as e:
            print(f"Error extracting text from PDF: {e}")
            return ""
    
    def _extract_from_image(self, image_path: Path) -> str:
        """
        Extract text from image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            Extracted text
        """
        try:
            print(f"\n📷 Reading image: {image_path}")
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
            
            print(f"   Image size: {image.shape[1]}x{image.shape[0]}")
            
            # Preprocess image
            print(f"   Preprocessing...")
            processed_image = self._preprocess_image(image)
            
            # Extract text
            print(f"   Running OCR with {self.engine}...")
            text = self._run_ocr(processed_image)
            
            print(f"\n✅ EXTRACTION COMPLETE")
            print(f"   Characters extracted: {len(text)}")
            print(f"{'='*60}\n")
            
            return text
        
        except Exception as e:
            print(f"\n❌ ERROR in image extraction: {e}")
            import traceback
            traceback.print_exc()
            print(f"{'='*60}\n")
            return ""
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Enhanced preprocessing for medical reports with multiple techniques
        
        Args:
            image: Input image array
            
        Returns:
            Preprocessed image optimized for Tesseract
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize if image is too small (Tesseract works best with 300 DPI)
            height, width = gray.shape
            if height < 1000 or width < 1000:
                scale_factor = max(1000 / height, 1000 / width, 1.5)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                print(f"      ↗ Upscaled: {width}x{height} → {new_width}x{new_height}")
            
            # Apply denoising
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            print(f"      ✓ Denoised")
            
            # Increase contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            contrast = clahe.apply(denoised)
            print(f"      ✓ Contrast enhanced")
            
            # Apply Otsu's thresholding for better binarization
            _, thresh = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            print(f"      ✓ Binarized (Otsu)")
            
            # Morphological operations to clean up
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
            print(f"      ✓ Morphological cleanup")
            
            return processed
            
        except Exception as e:
            print(f"      ⚠️ Preprocessing error: {e}, using original grayscale")
            # Return grayscale as fallback
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image
    
    def _run_ocr(self, image: np.ndarray) -> str:
        """
        Run OCR on preprocessed image
        
        Args:
            image: Preprocessed image
            
        Returns:
            Extracted text
        """
        if self.engine == 'tesseract':
            return self._tesseract_ocr(image)
        elif self.engine == 'easyocr':
            return self._easyocr_ocr(image)
        else:
            raise ValueError(f"Unsupported OCR engine: {self.engine}")
    
    def _tesseract_ocr(self, image: np.ndarray) -> str:
        """
        Extract text using Tesseract with multiple attempts and fallback strategies
        
        Args:
            image: Input image
            
        Returns:
            Extracted text
        """
        try:
            # Verify Tesseract is available
            tesseract_path = pytesseract.pytesseract.tesseract_cmd
            print(f"      Tesseract cmd: {tesseract_path}")
            
            # First attempt with configured PSM mode
            print(f"      Running with config: {self.config}")
            text = pytesseract.image_to_string(
                image,
                lang=self.language,
                config=self.config
            )
            
            chars_extracted = len(text.strip())
            print(f"      Result: {chars_extracted} characters")
            
            # If text is too short, try alternative PSM modes
            if chars_extracted < 50:
                print(f"      ⚠️ Low text detected ({chars_extracted} chars), trying alternative modes...")
                
                # PSM 6: Assume a single uniform block of text
                alt_config = '--oem 3 --psm 6'
                print(f"      Trying PSM 6...")
                text_alt = pytesseract.image_to_string(image, lang=self.language, config=alt_config)
                
                if len(text_alt.strip()) > chars_extracted:
                    text = text_alt
                    chars_extracted = len(text.strip())
                    print(f"      ✓ Better result: {chars_extracted} characters")
                
                # PSM 11: Sparse text (if still low)
                if chars_extracted < 50:
                    alt_config = '--oem 3 --psm 11'
                    print(f"      Trying PSM 11...")
                    text_alt = pytesseract.image_to_string(image, lang=self.language, config=alt_config)
                    
                    if len(text_alt.strip()) > chars_extracted:
                        text = text_alt
                        chars_extracted = len(text.strip())
                        print(f"      ✓ Better result: {chars_extracted} characters")
            
            result = text.strip()
            return result
            
        except pytesseract.TesseractNotFoundError as e:
            print(f"      ❌ TESSERACT NOT FOUND ERROR")
            print(f"         {e}")
            print(f"         Current path: {pytesseract.pytesseract.tesseract_cmd}")
            print(f"         Please install Tesseract or check the path")
            return ""
        except Exception as e:
            print(f"      ❌ Tesseract OCR error: {e}")
            print(f"         Tesseract path: {pytesseract.pytesseract.tesseract_cmd}")
            import traceback
            traceback.print_exc()
            return ""
    
    def _easyocr_ocr(self, image: np.ndarray) -> str:
        """
        Extract text using EasyOCR
        
        Args:
            image: Input image
            
        Returns:
            Extracted text
        """
        if self.easyocr_reader is None:
            return self._tesseract_ocr(image)
        
        try:
            results = self.easyocr_reader.readtext(image)
            text = " ".join([result[1] for result in results])
            return text.strip()
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return ""
    
    def extract_tables(self, file_path: Union[str, Path]) -> List[List[str]]:
        """
        Extract tables from medical reports (future enhancement)
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of extracted tables
        """
        # Placeholder for table extraction
        # Can be implemented using table detection models
        return []
    
    def get_confidence_scores(self, image: np.ndarray) -> dict:
        """
        Get OCR confidence scores
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with confidence metrics
        """
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self.language,
                output_type=pytesseract.Output.DICT
            )
            
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            
            if not confidences:
                return {'average_confidence': 0, 'text_detected': False}
            
            return {
                'average_confidence': sum(confidences) / len(confidences),
                'min_confidence': min(confidences),
                'max_confidence': max(confidences),
                'text_detected': True
            }
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return {'average_confidence': 0, 'text_detected': False}


def main():
    """Test OCR reader"""
    print("OCR Reader Test")
    print("=" * 50)
    
    reader = OCRReader()
    
    # Example usage
    print("\nOCR Reader initialized successfully!")
    print(f"Engine: {reader.engine}")
    print(f"Language: {reader.language}")
    
    # Test with a sample file (if exists)
    test_file = Path("test_report.pdf")
    if test_file.exists():
        print(f"\nExtracting text from {test_file}...")
        text = reader.extract_text(test_file)
        print(f"\nExtracted {len(text)} characters")
        print(f"Preview: {text[:200]}...")


if __name__ == "__main__":
    main()
