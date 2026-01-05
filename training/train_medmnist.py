"""
Train ViT Classifier on MedMNIST Dataset
High-quality medical image classification training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import timm
from pathlib import Path
import json
from tqdm import tqdm
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from backend.config import Config

try:
    import medmnist
    from medmnist import BloodMNIST, PathMNIST, ChestMNIST
except ImportError:
    print("Installing medmnist...")
    import os
    os.system(f"{sys.executable} -m pip install medmnist")
    import medmnist
    from medmnist import BloodMNIST, PathMNIST, ChestMNIST


class ViTMedicalClassifier(nn.Module):
    """Vision Transformer for medical image classification"""
    
    def __init__(self, num_classes, model_name='vit_base_patch16_224'):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=num_classes
        )
    
    def forward(self, x):
        return self.model(x)


class MedMNISTTrainer:
    """Trainer for MedMNIST datasets"""
    
    def __init__(self, dataset_name='bloodmnist', num_epochs=10):
        """
        Initialize trainer
        
        Args:
            dataset_name: Name of MedMNIST dataset (bloodmnist, pathmnist, chestmnist)
            num_epochs: Number of training epochs
        """
        self.dataset_name = dataset_name
        self.num_epochs = num_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print(f"Training {dataset_name.upper()} with Vision Transformer")
        print(f"Device: {self.device}")
        print(f"{'='*60}\n")
        
        # Load dataset
        self.load_dataset()
        
        # Initialize model
        self.model = ViTMedicalClassifier(
            num_classes=self.num_classes,
            model_name='vit_base_patch16_224'
        ).to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.0001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def load_dataset(self):
        """Load MedMNIST dataset"""
        print(f"📥 Loading {self.dataset_name}...")
        
        # Dataset class mapping
        dataset_classes = {
            'bloodmnist': BloodMNIST,
            'pathmnist': PathMNIST,
            'chestmnist': ChestMNIST
        }
        
        DataClass = dataset_classes.get(self.dataset_name.lower(), BloodMNIST)
        
        # Transforms (MedMNIST already returns PIL Images, so no need for ToPILImage)
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        data_path = Config.DATASET_PATH / 'MedMNIST'
        
        train_dataset = DataClass(split='train', download=True, root=data_path, transform=self.train_transform)
        val_dataset = DataClass(split='val', download=True, root=data_path, transform=self.val_transform)
        test_dataset = DataClass(split='test', download=True, root=data_path, transform=self.val_transform)
        
        # Get number of classes
        info = medmnist.INFO[self.dataset_name]
        self.num_classes = len(info['label'])
        self.class_names = info['label']
        
        # Create data loaders
        batch_size = 64 if self.dataset_name == 'bloodmnist' else 32
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Train: {len(train_dataset)} samples")
        print(f"   Val:   {len(val_dataset)} samples")
        print(f"   Test:  {len(test_dataset)} samples")
        print(f"   Classes: {self.num_classes} ({', '.join(self.class_names)})")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)
            
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
                'loss': f'{running_loss/len(pbar):.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation')
            for images, labels in pbar:
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{running_loss/len(pbar):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self):
        """Main training loop"""
        print(f"\n🚀 Starting training for {self.num_epochs} epochs...")
        best_val_acc = 0.0
        
        for epoch in range(self.num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\n   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(f'best_model_{self.dataset_name}.pth')
                print(f"   ✅ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print(f"\n{'='*60}")
        print(f"🎉 Training completed!")
        print(f"   Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"{'='*60}")
        
        # Test the model
        self.test()
        
        return self.history
    
    def test(self):
        """Test the model"""
        print(f"\n🧪 Testing model on test set...")
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc='Testing'):
                images = images.to(self.device)
                labels = labels.squeeze().long().to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        test_acc = 100. * correct / total
        print(f"\n✅ Test Accuracy: {test_acc:.2f}%")
        
        return test_acc
    
    def save_model(self, filename='vit_classifier.pth'):
        """Save model checkpoint"""
        save_path = Config.MODEL_PATH / 'vit_classifier' / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'dataset_name': self.dataset_name,
            'history': self.history
        }
        
        torch.save(checkpoint, save_path)
        print(f"💾 Model saved to {save_path}")


def main():
    """Main function"""
    print("\n🏥 Medical Image Classifier Training")
    print("Using MedMNIST datasets for high-quality medical image classification\n")
    
    # Choose dataset
    datasets = ['bloodmnist', 'pathmnist', 'chestmnist']
    print("Available datasets:")
    for i, name in enumerate(datasets, 1):
        print(f"  {i}. {name}")
    
    choice = input("\nSelect dataset (1-3) or press Enter for bloodmnist: ").strip()
    
    if choice == '2':
        dataset_name = 'pathmnist'
        epochs = 15  # PathMNIST has more samples
    elif choice == '3':
        dataset_name = 'chestmnist'
        epochs = 15
    else:
        dataset_name = 'bloodmnist'
        epochs = 10
    
    # Train model
    trainer = MedMNISTTrainer(dataset_name=dataset_name, num_epochs=epochs)
    history = trainer.train()
    
    # Save training history
    history_path = Config.MODEL_PATH / 'vit_classifier' / f'training_history_{dataset_name}.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"\n📈 Training history saved to {history_path}")


if __name__ == "__main__":
    main()
