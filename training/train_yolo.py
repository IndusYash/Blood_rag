"""
YOLO Detector Training Script
Trains a YOLO model for detecting tables/regions in medical reports
"""

from ultralytics import YOLO
from pathlib import Path
import yaml
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import Config


class YOLOTrainer:
    """YOLO model trainer for medical report object detection"""
    
    def __init__(self, model_size='n'):
        """
        Initialize YOLO trainer
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.model_size = model_size
        self.model_name = f'yolov8{model_size}.pt'
        self.model = None
        
        print(f"Initializing YOLOv8{model_size} trainer")
    
    def create_data_yaml(self, dataset_path, class_names):
        """
        Create YOLO data configuration file
        
        Args:
            dataset_path: Path to dataset
            class_names: List of class names
        """
        data_config = {
            'path': str(dataset_path),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': len(class_names),
            'names': class_names
        }
        
        yaml_path = dataset_path / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"Data configuration saved to: {yaml_path}")
        return yaml_path
    
    def train(
        self,
        data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640,
        save_dir=None
    ):
        """
        Train YOLO model
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
            save_dir: Directory to save results
        """
        # Load model
        self.model = YOLO(self.model_name)
        
        # Set save directory
        if save_dir is None:
            save_dir = Config.MODEL_PATH / 'yolo_detector'
        
        print(f"\nStarting YOLO training...")
        print(f"Model: {self.model_name}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Image size: {imgsz}")
        print("-" * 60)
        
        # Train
        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            project=str(save_dir),
            name='train',
            patience=20,
            save=True,
            device=0 if Config.DEBUG else 'cpu',
            workers=4,
            augment=True
        )
        
        print("\nTraining completed!")
        return results
    
    def validate(self, data_yaml):
        """
        Validate trained model
        
        Args:
            data_yaml: Path to data YAML file
        """
        if self.model is None:
            print("No model loaded. Train first or load a checkpoint.")
            return
        
        print("\nValidating model...")
        results = self.model.val(data=str(data_yaml))
        
        print("\nValidation Results:")
        print(f"mAP50: {results.box.map50:.4f}")
        print(f"mAP50-95: {results.box.map:.4f}")
        
        return results
    
    def export(self, format='onnx'):
        """
        Export model to different formats
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite')
        """
        if self.model is None:
            print("No model loaded.")
            return
        
        print(f"\nExporting model to {format}...")
        self.model.export(format=format)
        print("Export completed!")


def setup_dataset_structure():
    """
    Create dataset directory structure for YOLO
    """
    dataset_path = Config.DATASET_PATH / 'medical_tables'
    
    # Create directories
    directories = [
        'images/train',
        'images/val',
        'images/test',
        'labels/train',
        'labels/val',
        'labels/test'
    ]
    
    for directory in directories:
        dir_path = dataset_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Dataset structure created at: {dataset_path}")
    print("\nDirectory structure:")
    for directory in directories:
        print(f"  - {directory}/")
    
    print("\nInstructions:")
    print("1. Place training images in images/train/")
    print("2. Place validation images in images/val/")
    print("3. Place corresponding YOLO format labels in labels/train/ and labels/val/")
    print("4. Label format: <class_id> <x_center> <y_center> <width> <height>")
    
    return dataset_path


def main():
    """Main training function"""
    print("=" * 60)
    print("YOLO Training for Medical Report Table Detection")
    print("=" * 60)
    
    # Setup dataset structure
    print("\n1. Setting up dataset structure...")
    dataset_path = setup_dataset_structure()
    
    # Check if dataset exists
    train_images = list((dataset_path / 'images' / 'train').glob('*'))
    if not train_images:
        print("\nWarning: No training images found!")
        print(f"Please add images to: {dataset_path}/images/train/")
        print("And labels to: {dataset_path}/labels/train/")
        return
    
    print(f"\nFound {len(train_images)} training images")
    
    # Create data.yaml
    print("\n2. Creating data configuration...")
    class_names = ['table', 'text_block', 'header', 'footer']  # Customize as needed
    data_yaml = YOLOTrainer('n').create_data_yaml(dataset_path, class_names)
    
    # Initialize trainer
    print("\n3. Initializing YOLO trainer...")
    trainer = YOLOTrainer(model_size='n')  # Use 'n' for nano, 's' for small, etc.
    
    # Train model
    print("\n4. Starting training...")
    results = trainer.train(
        data_yaml=data_yaml,
        epochs=100,
        batch_size=16,
        imgsz=640
    )
    
    # Validate
    print("\n5. Validating model...")
    trainer.validate(data_yaml)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("Best model saved at: backend/models/yolo_detector/train/weights/best.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
