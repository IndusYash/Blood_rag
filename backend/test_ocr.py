"""
MINIMAL TESSERACT OCR TEST
Pure OCR extraction - no Flask, no analysis, just testing Tesseract
"""

import pytesseract
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import sys

try:
    from pdf2image import convert_from_path
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("⚠️ pdf2image not installed. PDF support disabled.")
    print("   Install: pip install pdf2image")

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("\n" + "="*70)
print("TESSERACT OCR TEST - MEDICAL REPORT EXTRACTION")
print("="*70)

def test_tesseract():
    """Test if Tesseract is working"""
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract installed: version {version}")
        print(f"   Path: {pytesseract.pytesseract.tesseract_cmd}\n")
        return True
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        print(f"   Install from: https://github.com/UB-Mannheim/tesseract/wiki\n")
        return False


def preprocess_image(image_path: str, save_debug: bool = True):
    """
    Preprocess image with multiple techniques
    """
    print(f"\n📷 Loading: {image_path}")
    
    # Check if PDF
    if image_path.lower().endswith('.pdf'):
        if not PDF_SUPPORT:
            print("❌ PDF support not available. Install pdf2image and poppler")
            return None
        
        try:
            print("   Converting PDF to image...")
            # Convert first page of PDF to image
            images = convert_from_path(image_path, dpi=300, first_page=1, last_page=1)
            if not images:
                print("❌ No pages in PDF")
                return None
            
            # Convert PIL Image to OpenCV format
            img = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
            print(f"   ✅ PDF converted to image")
        except Exception as e:
            print(f"❌ Could not convert PDF: {e}")
            return None
    else:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Could not read image: {image_path}")
            return None
    
    print(f"   Original size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Get base name for saving
    base_name = Path(image_path).stem
    debug_folder = Path("ocr_debug")
    debug_folder.mkdir(exist_ok=True)
    
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_1_grayscale.jpg"), gray)
        print(f"   Saved: {base_name}_1_grayscale.jpg")
    
    # Method 1: Simple threshold
    print("\n🔧 Trying Method 1: Simple Binary Threshold")
    _, thresh1 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_2_simple_thresh.jpg"), thresh1)
    
    # Method 2: Otsu's thresholding
    print("🔧 Trying Method 2: Otsu's Thresholding")
    _, thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_3_otsu_thresh.jpg"), thresh2)
    
    # Method 3: Adaptive threshold
    print("🔧 Trying Method 3: Adaptive Threshold")
    thresh3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_4_adaptive_thresh.jpg"), thresh3)
    
    # Method 4: Upscale + denoise + Otsu
    print("🔧 Trying Method 4: Enhanced (Upscale + Denoise + Otsu)")
    
    # Upscale if too small
    height, width = gray.shape
    if height < 1500 or width < 1500:
        scale = max(1500 / height, 1500 / width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        gray_scaled = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        print(f"   Upscaled to: {new_width}x{new_height}")
    else:
        gray_scaled = gray
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray_scaled, h=10)
    
    # Contrast enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_5_enhanced.jpg"), enhanced)
    
    # Final threshold
    _, thresh4 = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if save_debug:
        cv2.imwrite(str(debug_folder / f"{base_name}_6_final.jpg"), thresh4)
    
    print(f"\n✅ All preprocessed images saved to: {debug_folder}/")
    
    return {
        'original': gray,
        'simple_thresh': thresh1,
        'otsu': thresh2,
        'adaptive': thresh3,
        'enhanced': thresh4
    }


def extract_text_multiple_methods(images: dict, config: str = '--oem 3 --psm 3'):
    """
    Try OCR on all preprocessed versions
    """
    print(f"\n{'='*70}")
    print("RUNNING TESSERACT OCR")
    print(f"Config: {config}")
    print(f"{'='*70}\n")
    
    results = {}
    
    for method_name, img in images.items():
        print(f"\n🔍 Testing: {method_name}")
        print(f"   Image size: {img.shape[1]}x{img.shape[0]}")
        
        try:
            # Extract text
            text = pytesseract.image_to_string(img, lang='eng', config=config)
            char_count = len(text.strip())
            word_count = len(text.split())
            
            print(f"   ✅ Extracted: {char_count} characters, {word_count} words")
            
            # Show first 200 chars
            if char_count > 0:
                preview = text[:200].replace('\n', ' ')
                print(f"   Preview: {preview}...")
            
            results[method_name] = {
                'text': text,
                'char_count': char_count,
                'word_count': word_count
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[method_name] = {'text': '', 'char_count': 0, 'word_count': 0}
    
    return results


def try_different_psm_modes(best_image, image_name="image"):
    """
    Try different PSM (Page Segmentation Mode) settings
    """
    print(f"\n{'='*70}")
    print("TRYING DIFFERENT PSM MODES")
    print(f"{'='*70}\n")
    
    psm_modes = {
        '3': 'Fully automatic page segmentation (default)',
        '6': 'Uniform block of text',
        '11': 'Sparse text',
        '1': 'Automatic with OSD (Orientation and Script Detection)',
        '4': 'Single column of text',
    }
    
    results = {}
    
    for psm, description in psm_modes.items():
        config = f'--oem 3 --psm {psm}'
        print(f"\n🔍 PSM Mode {psm}: {description}")
        print(f"   Config: {config}")
        
        try:
            text = pytesseract.image_to_string(best_image, lang='eng', config=config)
            char_count = len(text.strip())
            word_count = len(text.split())
            
            print(f"   ✅ Result: {char_count} chars, {word_count} words")
            
            if char_count > 0:
                preview = text[:150].replace('\n', ' ')
                print(f"   Preview: {preview}...")
            
            results[f'PSM_{psm}'] = {
                'text': text,
                'char_count': char_count,
                'word_count': word_count,
                'config': config
            }
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[f'PSM_{psm}'] = {'text': '', 'char_count': 0, 'word_count': 0}
    
    return results


def main():
    """Main test function"""
    
    # Test Tesseract installation
    if not test_tesseract():
        return
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nUsage: python test_ocr.py <image_path>")
        print("\nLooking for test images in uploads folder...")
        
        uploads_folder = Path("uploads")
        if uploads_folder.exists():
            images = list(uploads_folder.glob("*.*"))
            if images:
                image_path = str(images[0])
                print(f"Found: {image_path}")
            else:
                print("❌ No images found in uploads folder")
                print("\nPlace a blood report image/PDF in backend/uploads/ or specify path:")
                print("   python test_ocr.py path/to/your/report.jpg")
                return
        else:
            print("❌ No uploads folder found")
            print("\nUsage: python test_ocr.py path/to/your/report.jpg")
            return
    
    if not Path(image_path).exists():
        print(f"❌ File not found: {image_path}")
        return
    
    # Preprocess image with multiple methods
    preprocessed = preprocess_image(image_path, save_debug=True)
    
    if preprocessed is None:
        return
    
    # Try OCR on all methods
    results = extract_text_multiple_methods(preprocessed)
    
    # Find best result
    best_method = max(results.items(), key=lambda x: x[1]['char_count'])
    best_name, best_result = best_method
    
    print(f"\n{'='*70}")
    print(f"BEST RESULT: {best_name}")
    print(f"Characters: {best_result['char_count']}")
    print(f"Words: {best_result['word_count']}")
    print(f"{'='*70}")
    
    # Use the best preprocessed image to try different PSM modes
    if best_result['char_count'] > 0:
        print("\nTrying different PSM modes on best preprocessing method...")
        psm_results = try_different_psm_modes(preprocessed[best_name])
        
        # Find best PSM
        best_psm = max(psm_results.items(), key=lambda x: x[1]['char_count'])
        psm_name, psm_result = best_psm
        
        print(f"\n{'='*70}")
        print(f"BEST PSM MODE: {psm_name}")
        print(f"Characters: {psm_result['char_count']}")
        print(f"{'='*70}")
        
        # Save the best extracted text
        if psm_result['char_count'] > 0:
            output_file = Path("ocr_debug") / "extracted_text.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(psm_result['text'])
            print(f"\n💾 Full text saved to: {output_file}")
            
            # Show full text
            print(f"\n{'='*70}")
            print("FULL EXTRACTED TEXT:")
            print(f"{'='*70}")
            print(psm_result['text'])
            print(f"{'='*70}\n")
    else:
        print("\n⚠️ No text could be extracted from the image")
        print("\nPossible issues:")
        print("1. Image quality is too low")
        print("2. Text is too small or blurry")
        print("3. Image is rotated or skewed")
        print("4. Text language is not English")
        print("\nCheck the debug images in ocr_debug/ folder to see preprocessing results")


if __name__ == "__main__":
    main()
