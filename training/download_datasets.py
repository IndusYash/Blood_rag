"""
Dataset Downloader for Medical Report AI
Downloads and prepares medical imaging datasets
"""

import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import tarfile
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from backend.config import Config


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_medmnist():
    """Download MedMNIST datasets"""
    print("\n" + "="*60)
    print("Downloading MedMNIST Datasets")
    print("="*60)
    
    medmnist_path = Config.DATASET_PATH / 'MedMNIST'
    medmnist_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import medmnist
        from medmnist import INFO
        
        print("\nAvailable MedMNIST datasets:")
        datasets = ['pathmnist', 'chestmnist', 'dermamnist', 'octmnist', 'pneumoniamnist', 
                   'retinamnist', 'breastmnist', 'bloodmnist', 'tissuemnist', 'organamnist',
                   'organcmnist', 'organsmnist']
        
        # Download selected datasets most relevant for medical reports
        selected = ['bloodmnist', 'pathmnist', 'chestmnist']
        
        for dataset_name in selected:
            print(f"\n📥 Downloading {dataset_name}...")
            DataClass = getattr(medmnist, INFO[dataset_name]['python_class'])
            
            # Download train, validation, and test sets
            train_dataset = DataClass(split='train', download=True, root=medmnist_path)
            val_dataset = DataClass(split='val', download=True, root=medmnist_path)
            test_dataset = DataClass(split='test', download=True, root=medmnist_path)
            
            print(f"✓ {dataset_name}: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
        
        print("\n✅ MedMNIST datasets downloaded successfully!")
        return True
        
    except ImportError:
        print("\n⚠️  medmnist package not found. Installing...")
        os.system(f"{sys.executable} -m pip install medmnist")
        print("Please run this script again after installation.")
        return False


def download_sample_medical_reports():
    """Download sample medical report images for training"""
    print("\n" + "="*60)
    print("Setting Up Sample Medical Reports Dataset")
    print("="*60)
    
    custom_docs_path = Config.DATASET_PATH / 'custom_medical_docs'
    
    # Create directories for different report types
    report_types = [
        'blood_test',
        'xray_chest',
        'mri_brain',
        'ct_scan',
        'ultrasound',
        'ecg',
        'lab_report'
    ]
    
    for report_type in report_types:
        (custom_docs_path / report_type).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {report_type}/")
    
    print("\n📝 Dataset structure created!")
    print(f"\nTo train the classifier, add your medical report images to:")
    print(f"  {custom_docs_path}/")
    print("\nOrganize them by report type in the respective folders.")
    
    # Create a README
    readme_content = """# Custom Medical Documents Dataset

## Directory Structure

Place your medical report images in the appropriate folders:

- `blood_test/` - Blood test reports, CBC, lipid panels, etc.
- `xray_chest/` - Chest X-ray images
- `mri_brain/` - Brain MRI scans
- `ct_scan/` - CT scan images
- `ultrasound/` - Ultrasound images
- `ecg/` - ECG/EKG reports
- `lab_report/` - General laboratory reports

## Image Requirements

- Formats: JPG, PNG, JPEG
- Size: Recommended 224x224 or larger
- Quality: Clear, readable text
- Minimum: 50-100 images per category for good training

## Data Collection Tips

1. Use publicly available datasets (ensure licensing)
2. Anonymize any real patient data
3. Maintain consistent image quality
4. Balance the dataset (similar number of images per class)

## Training

Once you have enough images, run:
```bash
cd training
python train_vit.py
```
"""
    
    with open(custom_docs_path / 'README.md', 'w') as f:
        f.write(readme_content)
    
    return True


def download_nih_chest_xray():
    """Information about NIH Chest X-ray dataset"""
    print("\n" + "="*60)
    print("NIH Chest X-ray Dataset Information")
    print("="*60)
    
    print("""
The NIH Chest X-ray dataset is one of the largest publicly available:
- 112,120 frontal-view X-ray images
- 30,805 unique patients
- 14 disease labels
- Size: ~45GB

To download (requires Kaggle API):
1. Install kaggle: pip install kaggle
2. Set up Kaggle credentials
3. Run: kaggle datasets download -d nih-chest-xrays/data
4. Extract to: dataset/nih_chest_xray/

Alternative: Manual download from:
https://www.kaggle.com/datasets/nih-chest-xrays/data
""")


def download_skin_cancer_dataset():
    """Information about HAM10000 skin lesion dataset"""
    print("\n" + "="*60)
    print("HAM10000 Skin Lesion Dataset Information")
    print("="*60)
    
    print("""
HAM10000 (Human Against Machine with 10000 training images):
- 10,015 dermatoscopic images
- 7 categories of skin lesions
- Size: ~2GB

To download:
1. Visit: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
2. Download manually or use Kaggle API
3. Extract to: dataset/skin_cancer/

This is useful for dermatology reports and image classification.
""")


def create_synthetic_training_data():
    """Create synthetic medical report images for immediate training"""
    print("\n" + "="*60)
    print("Creating Synthetic Training Data")
    print("="*60)
    
    try:
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        
        custom_docs_path = Config.DATASET_PATH / 'custom_medical_docs'
        
        # Sample report templates
        templates = {
            'blood_test': [
                "COMPLETE BLOOD COUNT\n\nHemoglobin: 14.5 g/dL\nWBC: 7,200/µL\nPlatelets: 250,000/µL\nRBC: 4.8 M/µL",
                "LIPID PROFILE\n\nTotal Cholesterol: 180 mg/dL\nLDL: 110 mg/dL\nHDL: 55 mg/dL\nTriglycerides: 140 mg/dL",
                "BLOOD GLUCOSE\n\nFasting: 95 mg/dL\nHbA1c: 5.4%\nRandom: 125 mg/dL"
            ],
            'lab_report': [
                "LIVER FUNCTION TEST\n\nALT: 35 U/L\nAST: 28 U/L\nBilirubin: 0.8 mg/dL\nAlbumin: 4.2 g/dL",
                "KIDNEY FUNCTION\n\nCreatinine: 1.0 mg/dL\nBUN: 15 mg/dL\neGFR: 95 mL/min\nUric Acid: 5.5 mg/dL",
                "THYROID PROFILE\n\nTSH: 2.5 mIU/L\nT3: 120 ng/dL\nT4: 8.5 µg/dL"
            ]
        }
        
        samples_per_type = 30
        
        for report_type, texts in templates.items():
            report_dir = custom_docs_path / report_type
            
            for i in range(samples_per_type):
                # Create synthetic report image
                img = Image.new('RGB', (800, 600), color='white')
                draw = ImageDraw.Draw(img)
                
                # Add text
                text = texts[i % len(texts)]
                
                # Add variations
                y_offset = 50 + (i % 3) * 20
                
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                
                draw.text((50, y_offset), text, fill='black', font=font)
                
                # Add some noise for realism
                img_array = np.array(img)
                noise = np.random.normal(0, 5, img_array.shape).astype(np.uint8)
                img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)
                
                # Save
                img.save(report_dir / f'sample_{i+1:03d}.jpg', quality=95)
            
            print(f"✓ Created {samples_per_type} synthetic samples for {report_type}")
        
        print(f"\n✅ Synthetic training data created!")
        print(f"Total images: {samples_per_type * len(templates)}")
        return True
        
    except Exception as e:
        print(f"⚠️  Error creating synthetic data: {e}")
        return False


def main():
    """Main download function"""
    print("\n🏥 Medical Report AI - Dataset Downloader")
    print("="*60)
    
    # Create base dataset directory
    Config.DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    print("\nSelect datasets to download:")
    print("1. MedMNIST (Medical MNIST datasets) - Recommended")
    print("2. Sample Medical Reports Structure")
    print("3. Create Synthetic Training Data - Quick Start")
    print("4. Show NIH Chest X-ray Info")
    print("5. Show Skin Cancer Dataset Info")
    print("6. All of the above")
    
    choice = input("\nEnter your choice (1-6): ").strip()
    
    if choice in ['1', '6']:
        download_medmnist()
    
    if choice in ['2', '6']:
        download_sample_medical_reports()
    
    if choice in ['3', '6']:
        create_synthetic_training_data()
    
    if choice in ['4', '6']:
        download_nih_chest_xray()
    
    if choice in ['5', '6']:
        download_skin_cancer_dataset()
    
    print("\n" + "="*60)
    print("✅ Dataset setup complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the dataset structure")
    print("2. Add more medical reports if needed")
    print("3. Run: python training/train_vit.py")
    print("\nFor questions, check: docs/README.md")


if __name__ == "__main__":
    main()
