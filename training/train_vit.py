"""
ViT Classifier Training Script
Trains a Vision Transformer model for medical report classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(__file__).parent.parent))

from backend.config import Config


class MedicalReportDataset(Dataset):
    """Dataset class for medical report images"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Initialize dataset
        
        Args:
            image_paths: List of image file paths
            labels: List of corresponding labels
            transform: Optional image transformations
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label


class ViTClassifier:
    """Vision Transformer classifier for medical reports"""
    
    def __init__(self, num_classes, model_name='vit_base_patch16_224', pretrained=True):
        """
        Initialize ViT classifier
        
        Args:
            num_classes: Number of output classes
            model_name: ViT model variant
            pretrained: Use pretrained weights
        """
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pretrained ViT model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
        self.model = self.model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.EPOCHS
        )
        
        print(f"Model initialized on {self.device}")
        print(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_transforms(self, train=True):
        """
        Get data transformations
        
        Args:
            train: Training or validation transforms
            
        Returns:
            Composed transformations
        """
        if train:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader):
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc='Validation'):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, epochs):
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        best_val_acc = 0.0
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Learning rate scheduler
            self.scheduler.step()
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(Config.VIT_MODEL_PATH)
                print(f"✓ Best model saved with validation accuracy: {val_acc:.2f}%")
        
        return history
    
    def save_model(self, path):
        """Save model checkpoint"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def prepare_dataset():
    """
    Prepare dataset for training
    Replace this with your actual dataset preparation logic
    """
    # Example: Assuming images are organized in folders by class
    dataset_path = Config.DATASET_PATH / 'custom_medical_docs'
    
    if not dataset_path.exists():
        print(f"Warning: Dataset path not found: {dataset_path}")
        print("Please organize your data in the dataset folder")
        return None, None, None, None
    
    # Collect images and labels
    image_paths = []
    labels = []
    class_names = []
    
    for class_idx, class_folder in enumerate(sorted(dataset_path.iterdir())):
        if class_folder.is_dir():
            class_names.append(class_folder.name)
            for img_path in class_folder.glob('*.jpg'):
                image_paths.append(str(img_path))
                labels.append(class_idx)
    
    if not image_paths:
        print("No images found in dataset")
        return None, None, None, None
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    return train_paths, val_paths, train_labels, val_labels


def main():
    """Main training function"""
    print("=" * 60)
    print("Vision Transformer Training for Medical Report Classification")
    print("=" * 60)
    
    # Prepare dataset
    print("\n1. Preparing dataset...")
    train_paths, val_paths, train_labels, val_labels = prepare_dataset()
    
    if train_paths is None:
        print("\nDataset preparation failed. Please check your dataset folder.")
        return
    
    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")
    
    # Get number of classes
    num_classes = len(set(train_labels))
    print(f"Number of classes: {num_classes}")
    
    # Initialize classifier
    print("\n2. Initializing model...")
    classifier = ViTClassifier(num_classes=num_classes)
    
    # Create datasets
    print("\n3. Creating data loaders...")
    train_dataset = MedicalReportDataset(
        train_paths,
        train_labels,
        transform=classifier.get_transforms(train=True)
    )
    val_dataset = MedicalReportDataset(
        val_paths,
        val_labels,
        transform=classifier.get_transforms(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2
    )
    
    # Train model
    print("\n4. Starting training...")
    history = classifier.train(train_loader, val_loader, Config.EPOCHS)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
