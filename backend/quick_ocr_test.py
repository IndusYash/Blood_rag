"""
QUICK OCR TEST - Create a test image and extract text
"""

import pytesseract
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("\n" + "="*60)
print("CREATING TEST IMAGE FOR OCR")
print("="*60)

# Create a simple test image with text
img = Image.new('RGB', (800, 400), color='white')
draw = ImageDraw.Draw(img)

# Add test text (simulating blood report)
test_text = """
BLOOD TEST REPORT
Patient Name: John Doe
Date: 01/06/2026

TEST NAME         RESULT    UNIT      REFERENCE RANGE
WBC Count         8500      /µL       4000-11000
Hemoglobin        14.5      g/dL      13-17
RBC Count         4.8       million   4.5-5.5
Platelet Count    250000    /µL       150000-400000
Glucose           95        mg/dL     70-100
"""

# Draw text on image
y_position = 20
for line in test_text.strip().split('\n'):
    draw.text((20, y_position), line, fill='black')
    y_position += 25

# Save test image
img.save('test_image.jpg')
print("✅ Test image created: test_image.jpg")

# Convert to OpenCV format
img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

# Test OCR
print("\n🔍 Running Tesseract OCR...")
text = pytesseract.image_to_string(img_cv, lang='eng', config='--oem 3 --psm 6')

print("\n" + "="*60)
print("EXTRACTED TEXT:")
print("="*60)
print(text)
print("="*60)

char_count = len(text.strip())
word_count = len(text.split())

print(f"\n📊 Statistics:")
print(f"   Characters: {char_count}")
print(f"   Words: {word_count}")

if char_count > 50:
    print("\n✅ SUCCESS! Tesseract is working correctly!")
    print("\nThe issue might be with your PDF/image file:")
    print("1. Try uploading a clear, high-resolution image (JPG/PNG)")
    print("2. Make sure text is not too small or blurry")
    print("3. Install Poppler for PDF support:")
    print("   Download: https://github.com/oschwartz10612/poppler-windows/releases")
    print("   Add to PATH or place in your project folder")
else:
    print("\n⚠️ Tesseract is not extracting text properly")
    print("   Check Tesseract installation and configuration")
