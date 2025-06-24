import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import warnings
import gc  # Garbage collection
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class MemoryEfficientSkinCancerDataset(Dataset):
    """Memory-efficient dataset that loads images on-the-fly"""
    def __init__(self, image_paths, labels, dataset_path, image_size=(224, 224), transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image on-the-fly
        image_path = os.path.join(self.dataset_path, self.image_paths[idx])
        image = self.load_and_preprocess_image(image_path)
        label = self.labels[idx]
        
        if image is None:
            # Return a black image if loading fails
            image = np.zeros((*self.image_size, 3), dtype=np.float32)
        
        if self.transform:
            # Convert numpy array to PIL Image for transforms
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return image, torch.LongTensor([label])[0]
    
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            image = cv2.resize(image, self.image_size)
            
            # Apply CLAHE for contrast enhancement
            image = self.apply_clahe(image)
            
            # Apply Gaussian filter for noise reduction
            image = cv2.GaussianBlur(image, (5, 5), 1.0)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def apply_clahe(self, image):
        """Apply Contrast Limited Adaptive Histogram Equalization"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        return enhanced

class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class MemoryOptimizedSkinCancerPipeline:
    def __init__(self, dataset_path, image_size=(224, 224), batch_size=16, max_samples=2000):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.max_samples = max_samples  # Limit total samples
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Maximum samples limit: {self.max_samples}")
        
    def load_and_filter_data(self, metadata_path=None, positive_negative_ratio=10):
        """
        Load and filter dataset with memory optimization
        """
        print("Loading and filtering dataset...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Limit the number of files to process
        if len(image_files) > self.max_samples:
            print(f"Limiting dataset to {self.max_samples} images for memory efficiency")
            image_files = image_files[:self.max_samples]
        
        if metadata_path and os.path.exists(metadata_path):
            # If metadata exists, use it for proper labeling
            metadata = pd.read_csv(metadata_path)
            # Filter metadata to only include available images
            metadata = metadata[metadata['image_name'].isin(image_files)]
            filtered_data = self.filter_by_ratio(metadata, positive_negative_ratio)
        else:
            # Create dummy labels for demonstration
            print("Warning: No metadata provided. Creating dummy labels for demonstration.")
            print("Please provide actual metadata file with image names and labels.")
            
            # Create balanced labels
            n_positive = min(100, len(image_files) // (positive_negative_ratio + 1))
            n_negative = min(n_positive * positive_negative_ratio, len(image_files) - n_positive)
            
            labels = [1] * n_positive + [0] * n_negative
            selected_files = image_files[:len(labels)]
            
            data = pd.DataFrame({
                'image_name': selected_files,
                'target': labels
            })
            filtered_data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return filtered_data
    
    def filter_by_ratio(self, data, ratio):
        """Filter data to maintain positive:negative ratio"""
        positive_samples = data[data['target'] == 1]
        negative_samples = data[data['target'] == 0]
        
        # Limit positive samples
        max_positive = min(100, len(positive_samples))
        if len(positive_samples) > max_positive:
            positive_samples = positive_samples.sample(n=max_positive, random_state=42)
        
        # Calculate required negative samples
        required_negative = len(positive_samples) * ratio
        max_negative = min(required_negative, len(negative_samples), self.max_samples - len(positive_samples))
        
        if len(negative_samples) > max_negative:
            negative_samples = negative_samples.sample(n=int(max_negative), random_state=42)
        
        filtered_data = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        print(f"Filtered dataset: {len(positive_samples)} positive, {len(negative_samples)} negative samples")
        
        return filtered_data
    
    def apply_smart_smote(self, image_paths, labels):
        """
        Apply SMOTE only on a small subset to reduce memory usage
        """
        print("Applying smart SMOTE for data balancing...")
        
        # Convert to arrays
        labels = np.array(labels)
        
        # Count classes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"Original class distribution: {dict(zip(unique, counts))}")
        
        # Only apply SMOTE if we have significant class imbalance
        if len(unique) > 1 and min(counts) / max(counts) < 0.5:
            # Load a small subset of images for SMOTE feature extraction
            print("Loading subset for SMOTE...")
            subset_size = min(200, len(image_paths))
            indices = np.random.choice(len(image_paths), subset_size, replace=False)
            
            # Load images for feature extraction
            features = []
            subset_labels = []
            
            for idx in indices:
                image_path = os.path.join(self.dataset_path, image_paths[idx])
                image = cv2.imread(image_path)
                if image is not None:
                    # Simple feature extraction - mean color values
                    image = cv2.resize(image, (32, 32))  # Very small for memory efficiency
                    feature = image.mean(axis=(0, 1))  # RGB means
                    features.append(feature)
                    subset_labels.append(labels[idx])
            
            if len(features) > 0:
                features = np.array(features)
                subset_labels = np.array(subset_labels)
                
                # Apply SMOTE on features
                if len(np.unique(subset_labels)) > 1:
                    try:
                        n_neighbors = min(3, min(np.bincount(subset_labels)) - 1)
                        if n_neighbors > 0:
                            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
                            X_resampled, y_resampled = smote.fit_resample(features, subset_labels)
                            
                            # Calculate how many synthetic samples were created
                            synthetic_count = len(y_resampled) - len(subset_labels)
                            print(f"SMOTE would create {synthetic_count} synthetic samples")
                            
                            # For simplicity, just duplicate existing minority samples
                            minority_class = unique[np.argmin(counts)]
                            minority_indices = np.where(labels == minority_class)[0]
                            
                            # Duplicate minority samples
                            duplicates_needed = min(synthetic_count, len(minority_indices))
                            if duplicates_needed > 0:
                                duplicate_indices = np.random.choice(minority_indices, duplicates_needed, replace=True)
                                
                                # Add duplicates
                                enhanced_paths = image_paths + [image_paths[i] for i in duplicate_indices]
                                enhanced_labels = labels.tolist() + [labels[i] for i in duplicate_indices]
                                
                                print(f"Enhanced dataset: {len(enhanced_paths)} samples")
                                return enhanced_paths, enhanced_labels
                    
                    except Exception as e:
                        print(f"SMOTE failed: {e}, using original data")
        
        return image_paths, labels.tolist()
    
    def create_model(self, model_name='AlexNet', num_classes=2):
        """Create specified model with memory optimization"""
        if model_name == 'AlexNet':
            model = AlexNet(num_classes)
        elif model_name == 'ResNet-18':  # Use smaller ResNet
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            # Default to AlexNet
            model = AlexNet(num_classes)
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, epochs=20):
        """Train model with memory optimization"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1)
                    val_correct += (pred == target).sum().item()
                    val_total += target.size(0)
            
            val_acc = val_correct / val_total if val_total > 0 else 0
            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Memory cleanup
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()
        
        return model, best_val_acc, train_losses, val_accuracies
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model with memory optimization"""
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probs = F.softmax(output, dim=1)
                pred = output.argmax(dim=1)
                
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
                
                # Clear cache
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='binary') if len(set(all_targets)) > 1 else 0
        
        # Calculate sensitivity and specificity
        if len(set(all_targets)) > 1:
            cm = confusion_matrix(all_targets, all_preds)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = specificity = 0
        else:
            sensitivity = specificity = 0
        
        try:
            auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        except:
            auc = 0.5
        
        return {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc
        }
    
    def plot_results(self, results_dict):
        """Plot comparison of model performances"""
        if not results_dict:
            print("No results to plot")
            return
            
        models = list(results_dict.keys())
        metrics = ['accuracy', 'sensitivity', 'specificity', 'f1_score', 'auc']
        
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            values = [results_dict[model][metric] for model in models]
            bars = plt.bar(models, values)
            plt.title(f'{metric.replace("_", " ").title()}')
            plt.ylabel('Score')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                plt.text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self, metadata_path=None):
        """Run the complete pipeline with memory optimization"""
        print("Starting Memory-Optimized Skin Cancer Detection Pipeline...")
        
        try:
            # Step 1: Load and filter data
            data_df = self.load_and_filter_data(metadata_path)
            
            if len(data_df) == 0:
                print("No data loaded. Please check your dataset.")
                return {}
            
            print(f"Dataset size: {len(data_df)} samples")
            
            # Step 2: Apply smart SMOTE (memory-efficient)
            enhanced_paths, enhanced_labels = self.apply_smart_smote(
                data_df['image_name'].tolist(), 
                data_df['target'].tolist()
            )
            
            # Step 3: Split data
            train_paths, test_paths, train_labels, test_labels = train_test_split(
                enhanced_paths, enhanced_labels, test_size=0.2, random_state=42,
                stratify=enhanced_labels if len(set(enhanced_labels)) > 1 else None
            )
            
            # Further split training data for validation
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels, test_size=0.2, random_state=42,
                stratify=train_labels if len(set(train_labels)) > 1 else None
            )
            
            print(f"Training set: {len(train_paths)} samples")
            print(f"Validation set: {len(val_paths)} samples")
            print(f"Test set: {len(test_paths)} samples")
            
            # Step 4: Create memory-efficient datasets
            train_dataset = MemoryEfficientSkinCancerDataset(
                train_paths, train_labels, self.dataset_path, self.image_size
            )
            val_dataset = MemoryEfficientSkinCancerDataset(
                val_paths, val_labels, self.dataset_path, self.image_size
            )
            test_dataset = MemoryEfficientSkinCancerDataset(
                test_paths, test_labels, self.dataset_path, self.image_size
            )
            
            # Create data loaders with reduced batch size
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            
            # Step 5: Train models
            model_names = ['AlexNet', 'ResNet-18']
            results = {}
            
            for model_name in model_names:
                try:
                    print(f"\nTraining {model_name}...")
                    model = self.create_model(model_name)
                    
                    # Train model
                    trained_model, best_acc, train_losses, val_accs = self.train_model(
                        model, train_loader, val_loader, epochs=15
                    )
                    
                    # Evaluate on test set
                    test_results = self.evaluate_model(trained_model, test_loader)
                    results[model_name] = test_results
                    
                    print(f"{model_name} Results:")
                    for metric, value in test_results.items():
                        print(f"  {metric}: {value:.4f}")
                    
                    # Clear memory
                    del model, trained_model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()
                        
                except Exception as e:
                    print(f"Error training {model_name}: {str(e)}")
                    continue
            
            # Step 6: Plot results
            if results:
                self.plot_results(results)
                
                # Print final comparison
                print("\n" + "="*50)
                print("FINAL MODEL COMPARISON")
                print("="*50)
                
                for model_name, scores in results.items():
                    print(f"\n{model_name}:")
                    for metric, score in scores.items():
                        print(f"  {metric}: {score:.4f}")
            
            return results
            
        except Exception as e:
            print(f"Error in pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}

# Usage example with memory optimization
if __name__ == "__main__":
    # Initialize pipeline with memory constraints
    dataset_path = r"C:\Users\erand\OneDrive - University of Jaffna\6sem\research\isic-2024-challenge\train-image\image"
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Dataset path does not exist: {dataset_path}")
        print("Please check your dataset path.")
        exit()
    
    # Count images in dataset
    image_files = [f for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(image_files)} images in dataset")
    
    if len(image_files) == 0:
        print("No images found in the dataset path.")
        exit()
    
    # Initialize with memory-optimized settings
    pipeline = MemoryOptimizedSkinCancerPipeline(
        dataset_path=dataset_path,
        image_size=(128, 128),  # Reduced image size
        batch_size=8,          # Reduced batch size
        max_samples=1000       # Limit total samples
    )
    
    # Run complete pipeline
    results = pipeline.run_complete_pipeline(metadata_path=None)
    
    if results:
        print("\nPipeline completed successfully!")
        print("Check 'model_comparison.png' for visual results.")
    else:
        print("Pipeline failed to complete.")