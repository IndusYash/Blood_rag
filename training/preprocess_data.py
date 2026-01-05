"""
Data Preprocessing Script
Converts and prepares images for training
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import json
from tqdm import tqdm
import shutil
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import Config


class DataPreprocessor:
    """Preprocess medical report images for training"""
    
    def __init__(self, input_dir, output_dir):
        """
        Initialize preprocessor
        
        Args:
            input_dir: Input directory with raw images
            output_dir: Output directory for processed images
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'failed': 0,
            'skipped': 0
        }
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess a single image
        
        Args:
            image_path: Path to input image
            target_size: Target image size
            
        Returns:
            Preprocessed image
        """
        try:
            # Read image
            image = cv2.imread(str(image_path))
            
            if image is None:
                raise ValueError("Could not read image")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Denoise
            image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
            
            # Enhance contrast
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            image = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Resize
            image = cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            return image
        
        except Exception as e:
            print(f"Error preprocessing {image_path}: {e}")
            return None
    
    def process_directory(self, target_size=(224, 224)):
        """
        Process all images in directory
        
        Args:
            target_size: Target image size
        """
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(self.input_dir.rglob(ext))
        
        self.stats['total_files'] = len(image_files)
        
        if not image_files:
            print(f"No images found in {self.input_dir}")
            return
        
        print(f"Found {len(image_files)} images")
        print(f"Processing images to {target_size}...")
        
        # Process each image
        for img_path in tqdm(image_files, desc="Processing"):
            try:
                # Preprocess
                processed_image = self.preprocess_image(img_path, target_size)
                
                if processed_image is None:
                    self.stats['failed'] += 1
                    continue
                
                # Save processed image
                relative_path = img_path.relative_to(self.input_dir)
                output_path = self.output_dir / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert to PIL and save
                pil_image = Image.fromarray(processed_image)
                pil_image.save(output_path, quality=95)
                
                self.stats['processed'] += 1
            
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                self.stats['failed'] += 1
        
        self.print_stats()
    
    def augment_images(self, num_augmentations=5):
        """
        Create augmented versions of images
        
        Args:
            num_augmentations: Number of augmented versions per image
        """
        print(f"\nCreating {num_augmentations} augmented versions per image...")
        
        image_files = list(self.output_dir.rglob('*.jpg'))
        
        for img_path in tqdm(image_files, desc="Augmenting"):
            try:
                image = cv2.imread(str(img_path))
                
                for i in range(num_augmentations):
                    augmented = self._apply_augmentation(image)
                    
                    # Save augmented image
                    aug_path = img_path.parent / f"{img_path.stem}_aug{i}{img_path.suffix}"
                    cv2.imwrite(str(aug_path), augmented)
            
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")
    
    def _apply_augmentation(self, image):
        """
        Apply random augmentations to image
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        # Random rotation
        angle = np.random.uniform(-10, 10)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.8, 1.2)  # Contrast
        beta = np.random.uniform(-20, 20)     # Brightness
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # Random flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
        
        return image
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Split data into train/val/test sets
        
        Args:
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
        """
        print("\nCreating train/val/test splits...")
        
        # Get all images
        image_files = list(self.output_dir.rglob('*.jpg'))
        np.random.shuffle(image_files)
        
        # Calculate split indices
        n = len(image_files)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)
        
        # Split files
        train_files = image_files[:train_end]
        val_files = image_files[train_end:val_end]
        test_files = image_files[val_end:]
        
        # Create split directories
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        for split_name, files in splits.items():
            split_dir = self.output_dir.parent / f'split_{split_name}'
            split_dir.mkdir(exist_ok=True)
            
            print(f"\n{split_name.capitalize()}: {len(files)} images")
            
            for file in tqdm(files, desc=f"Copying {split_name}"):
                dest = split_dir / file.name
                shutil.copy2(file, dest)
    
    def print_stats(self):
        """Print processing statistics"""
        print("\n" + "=" * 50)
        print("Processing Statistics")
        print("=" * 50)
        print(f"Total files: {self.stats['total_files']}")
        print(f"Processed: {self.stats['processed']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped: {self.stats['skipped']}")
        print("=" * 50)


def main():
    """Main preprocessing function"""
    print("=" * 60)
    print("Medical Report Data Preprocessing")
    print("=" * 60)
    
    # Set paths
    input_dir = Config.DATASET_PATH / 'raw_images'
    output_dir = Config.DATASET_PATH / 'processed_images'
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nCreating input directory: {input_dir}")
        input_dir.mkdir(parents=True, exist_ok=True)
        print("\nPlease add raw medical report images to:")
        print(f"  {input_dir}")
        return
    
    # Initialize preprocessor
    print(f"\nInput directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    preprocessor = DataPreprocessor(input_dir, output_dir)
    
    # Process images
    print("\n1. Preprocessing images...")
    preprocessor.process_directory(target_size=(224, 224))
    
    # Optional: Create augmented versions
    create_augmentations = input("\nCreate augmented versions? (y/n): ").lower() == 'y'
    if create_augmentations:
        print("\n2. Creating augmentations...")
        preprocessor.augment_images(num_augmentations=3)
    
    # Optional: Create train/val/test splits
    create_splits = input("\nCreate train/val/test splits? (y/n): ").lower() == 'y'
    if create_splits:
        print("\n3. Creating splits...")
        preprocessor.create_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    print("\n" + "=" * 60)
    print("Preprocessing completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
