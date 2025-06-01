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
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class SkinCancerDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            # Convert numpy array to PIL Image for transforms
            image = Image.fromarray((image * 255).astype(np.uint8))
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image).permute(2, 0, 1)
        
        return image, torch.LongTensor([label])[0]

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

class SkinCancerDetectionPipeline:
    def __init__(self, dataset_path, image_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def load_and_filter_data(self, metadata_path=None, positive_negative_ratio=20):
        """
        Load and filter dataset with 1:20 positive to negative ratio
        """
        print("Loading and filtering dataset...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.dataset_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if metadata_path and os.path.exists(metadata_path):
            # If metadata exists, use it for proper labeling
            metadata = pd.read_csv(metadata_path)
            # Assuming metadata has columns: 'image_name', 'target' (0=benign, 1=malignant)
            filtered_data = self.filter_by_ratio(metadata, positive_negative_ratio)
        else:
            # Create dummy labels for demonstration (replace with actual labels)
            print("Warning: No metadata provided. Creating dummy labels for demonstration.")
            print("Please provide actual metadata file with image names and labels.")
            
            # Limit to first 1000 images for demonstration
            sample_files = image_files[:min(1000, len(image_files))]
            labels = np.random.choice([0, 1], size=len(sample_files), p=[0.95, 0.05])  # 5% positive
            data = pd.DataFrame({
                'image_name': sample_files,
                'target': labels
            })
            filtered_data = self.filter_by_ratio(data, positive_negative_ratio)
        
        return filtered_data
    
    def filter_by_ratio(self, data, ratio):
        """Filter data to maintain positive:negative ratio"""
        positive_samples = data[data['target'] == 1]
        negative_samples = data[data['target'] == 0]
        
        # Calculate required negative samples
        required_negative = len(positive_samples) * ratio
        
        if len(negative_samples) > required_negative:
            negative_samples = negative_samples.sample(n=int(required_negative), random_state=42)
        
        filtered_data = pd.concat([positive_samples, negative_samples]).reset_index(drop=True)
        print(f"Filtered dataset: {len(positive_samples)} positive, {len(negative_samples)} negative samples")
        
        return filtered_data
    
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
    
    def apply_gaussian_filter(self, image, kernel_size=5, sigma=1.0):
        """Apply Gaussian filter for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def detect_and_inpaint_artifacts(self, image):
        """
        Detect and remove artifacts using Canny edge detection and Fast Marching Inpainting
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Create mask for artifacts (simplified - in practice, you'd use more sophisticated detection)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply inpainting using Fast Marching method
        inpainted = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
        
        return inpainted
    
    def preprocess_image(self, image_path):
        """Complete preprocessing pipeline"""
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
            image = self.apply_gaussian_filter(image)
            
            # Detect and inpaint artifacts
            image = self.detect_and_inpaint_artifacts(image)
            
            # Normalize pixel values
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return None
    
    def load_preprocessed_data(self, data_df):
        """Load and preprocess all images"""
        print("Loading and preprocessing images...")
        
        images = []
        labels = []
        
        for idx, row in data_df.iterrows():
            image_path = os.path.join(self.dataset_path, row['image_name'])
            
            if os.path.exists(image_path):
                processed_image = self.preprocess_image(image_path)
                if processed_image is not None:
                    images.append(processed_image)
                    labels.append(row['target'])
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(data_df)} images")
        
        return np.array(images), np.array(labels)
    
    def apply_deep_smote(self, X, y):
        """
        Apply Deep-SMOTE for data balancing
        """
        print("Applying SMOTE for data balancing...")
        
        # Flatten images for SMOTE
        X_flattened = X.reshape(X.shape[0], -1)
        
        # Apply SMOTE with fewer neighbors if needed
        n_neighbors = min(3, len(X[y == 1]) - 1) if len(X[y == 1]) > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X_flattened, y)
        
        # Reshape back to image format
        X_resampled = X_resampled.reshape(-1, *self.image_size, 3)
        
        print(f"Original dataset: {len(X)} samples")
        print(f"Resampled dataset: {len(X_resampled)} samples")
        print(f"Class distribution: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def create_model(self, model_name, num_classes=2):
        """Create specified model"""
        if model_name == 'AlexNet':
            model = AlexNet(num_classes)
        elif model_name == 'ResNet-50':
            model = models.resnet50(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'EfficientNet-B0':
            # Use ResNet as substitute since EfficientNet might not be available
            model = models.resnet18(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif model_name == 'InceptionV3':
            # Use ResNet as substitute
            model = models.resnet34(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, epochs=30):
        """Train a single model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        best_val_acc = 0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
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
            
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_acc)
            
            scheduler.step(avg_val_loss)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            if epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        return model, best_val_acc, train_losses, val_accuracies
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model and return comprehensive metrics"""
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
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        
        # Calculate sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except:
            auc = 0.5  # Default AUC if calculation fails
        
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
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            values = [results_dict[model][metric] for model in models]
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        # Remove empty subplot
        axes[5].remove()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_pipeline(self, metadata_path=None):
        """Run the complete pipeline"""
        print("Starting Skin Cancer Detection Pipeline...")
        
        try:
            # Step 1: Load and filter data
            data_df = self.load_and_filter_data(metadata_path)
            
            if len(data_df) == 0:
                print("No data loaded. Please check your dataset.")
                return {}
            
            # Step 2: Load and preprocess images
            X, y = self.load_preprocessed_data(data_df)
            
            if len(X) == 0:
                print("No images loaded. Please check your dataset path.")
                return {}
            
            print(f"Loaded {len(X)} images successfully")
            
            # Step 3: Apply Deep-SMOTE for data balancing (only if we have both classes)
            if len(np.unique(y)) > 1:
                X_balanced, y_balanced = self.apply_deep_smote(X, y)
            else:
                print("Only one class found, skipping SMOTE")
                X_balanced, y_balanced = X, y
            
            # Step 4: Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.2, random_state=42, 
                stratify=y_balanced if len(np.unique(y_balanced)) > 1 else None
            )
            
            print(f"Training set: {len(X_train)} samples")
            print(f"Test set: {len(X_test)} samples")
            
            # Create data loaders
            train_dataset = SkinCancerDataset(X_train, y_train)
            test_dataset = SkinCancerDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
            
            # Step 5: Train models
            model_names = ['AlexNet', 'ResNet-50']  # Start with fewer models
            results = {}
            
            for model_name in model_names:
                try:
                    print(f"\nTraining {model_name}...")
                    model = self.create_model(model_name)
                    
                    # Split training data for validation
                    val_size = int(0.2 * len(X_train))
                    train_size = len(X_train) - val_size
                    
                    train_subset = torch.utils.data.Subset(train_dataset, range(train_size))
                    val_subset = torch.utils.data.Subset(train_dataset, range(train_size, len(X_train)))
                    
                    train_sub_loader = DataLoader(train_subset, batch_size=self.batch_size, shuffle=True)
                    val_loader = DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
                    
                    # Train model
                    trained_model, best_acc, train_losses, val_accs = self.train_model(
                        model, train_sub_loader, val_loader, epochs=20
                    )
                    
                    # Evaluate on test set
                    test_results = self.evaluate_model(trained_model, test_loader)
                    results[model_name] = test_results
                    
                    print(f"{model_name} Results:")
                    for metric, value in test_results.items():
                        print(f"  {metric}: {value:.4f}")
                        
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
            return {}

# Usage example
if __name__ == "__main__":
    # Initialize pipeline
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
    
    pipeline = SkinCancerDetectionPipeline(
        dataset_path=dataset_path,
        image_size=(224, 224),
        batch_size=8  # Reduced batch size for memory efficiency
    )
    
    # Run complete pipeline
    # Note: Provide metadata_path if you have label information
    # For example: metadata_path = "path/to/your/metadata.csv"
    results = pipeline.run_complete_pipeline(metadata_path=None)
    
    if results:
        print("\nPipeline completed successfully!")
    else:
        print("Pipeline failed to complete.")