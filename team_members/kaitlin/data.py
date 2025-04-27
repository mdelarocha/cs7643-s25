import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
import gcsfs
import nibabel as nib
import pandas as pd
from datetime import datetime
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.metrics import ConfusionMatrixDisplay
from IPython.display import display
from IPython.display import Image as IPythonImage


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize([0.45], [0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.45], [0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.45], [0.225])
])

# #CDR label mapping
cdr_label_map = {
    0: "nondemented",
    0.5: "very mild dementia",
    1: "mild to moderate dementia"  # Combined CDR 1.0 and 2.0
}

idx_to_cdr = {0: 0, 1: 0.5, 2: 1} 


class IdentityBlock(nn.Module):
    def __init__(self, dim):
        """
        dim: channel dimension of conv_layers
        """
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.bn1 = nn.InstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, groups=32, padding=1)
        self.bn2 = nn.InstanceNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.bn3 = nn.InstanceNorm2d(dim * 2)

    def forward(self, x):
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = F.relu(x)

        return x

class ProjectionBlock(nn.Module):
    def __init__(self, input_dim, dim, strides=1):
        """
        input_dim: input channel dimension
        dim: channel dimension of conv_layers
        strides: strides = 2 for conv3, conv4, conv5

        Projection block has 1x1 conv in the shortcut connection.
        """
        super(ProjectionBlock, self).__init__()
        self.shortcut = nn.Sequential(
            nn.Conv2d(input_dim, 2 * dim, kernel_size=1, stride=strides, bias=False),
            nn.InstanceNorm2d(2 * dim)
        )
        self.conv1 = nn.Conv2d(input_dim, dim, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=strides, padding=1, groups=32, bias=False)
        self.bn2 = nn.InstanceNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, 2 * dim, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.InstanceNorm2d(2 * dim)

    def forward(self, x):
        shortcut = self.shortcut(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        x += shortcut
        x = F.relu(x)

        return x

class ResNeXt50(nn.Module):
    def __init__(self, num_classes=3): 
        super(ResNeXt50, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2)  # Add dropout after stem
        )

        # Stage 1, 3 blocks
        self.stage1 = nn.Sequential(
            ProjectionBlock(64, 128, strides=1),
            IdentityBlock(128),
            IdentityBlock(128),
            nn.Dropout(0.2)  # Add dropout after stage1
        )

        # Stage 2, 4 blocks
        self.stage2 = nn.Sequential(
            ProjectionBlock(256, 256, strides=2),
            *[IdentityBlock(256) for _ in range(4)],
            nn.Dropout(0.3)  # Add dropout after stage2
        )

        # Stage 3, 6 blocks
        self.stage3 = nn.Sequential(
            ProjectionBlock(512, 512, strides=2),
            *[IdentityBlock(512) for _ in range(6)],
            nn.Dropout(0.4)  # Add dropout after stage3
        )

        # Stage 4, 3 blocks
        self.stage4 = nn.Sequential(
            ProjectionBlock(1024, 1024, strides=2),
            *[IdentityBlock(1024) for _ in range(3)],
            nn.Dropout(0.5)  # Add dropout after stage4
        )

        # Classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)  

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

#instantiate the model
resnext = ResNeXt50(num_classes=3)

#save model
torch.save(resnext.state_dict(), 'resnext50.pth')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_path = f'resnext50_backup_{timestamp}.pth'
torch.save(resnext.state_dict(), backup_path)


class MRIDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)  # Convert labels to tensor
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_subject_data(subject_path, fs):
    """Load MRI data for a single subject"""
    try:
        # Find the processed MPRAGE directory
        processed_path = f"{subject_path}/PROCESSED/MPRAGE/T88_111"
        if not fs.exists(processed_path):
            return None
            
        # Find the .img file
        img_files = [f for f in fs.ls(processed_path) if f.endswith('.img')]
        if not img_files:
            return None
            
        img_path = img_files[0]
        hdr_path = img_path.replace('.img', '.hdr')
        
        # Download files
        with fs.open(hdr_path, 'rb') as f:
            with open('temp.hdr', 'wb') as out_file:
                out_file.write(f.read())
        with fs.open(img_path, 'rb') as f:
            with open('temp.img', 'wb') as out_file:
                out_file.write(f.read())
                
        # Load the image
        img = nib.load('temp.img')
        return img.get_fdata()
    except Exception as e:
        print(f"Error loading subject {subject_path}: {str(e)}")
        return None

def prepare_data(df, transform=None):
    slices = []
    labels = []
    subject_ids = []  
    
    # Create mapping from CDR scores to integer indices
    cdr_to_idx = {0: 0, 0.5: 1, 1: 2, 2: 2}  # Map both CDR 1.0 and 2.0 to index 2
    
    # Define slice range
    slice_range = range(50, 120)
    
    # Create GCS filesystem
    fs = gcsfs.GCSFileSystem(token='anon')
    base_path = 'oasis-1-dataset-13635/oasis_raw/'
    
    # Process each disc
    for disc in range(1, 13):
        disc_path = f"{base_path}disc{disc}"
        print(f"\nProcessing disc {disc}")

        subjects = [s for s in fs.ls(disc_path)]
        
        for subject_path in subjects:
            #extract subject ID
            subject_id = subject_path.split('/')[-1]
            
            #match subject ID
            subject_data = df[df['ID'].str.contains(subject_id)]
            if subject_data.empty:
                continue
                
            cdr_score = subject_data['CDR'].iloc[0]
            label_idx = cdr_to_idx[cdr_score]
            
            #load MRI data
            img_data = load_subject_data(subject_path, fs)
            if img_data is None:
                continue
            
            #process slices
            valid_slices = 0
            subject_slices = []
            for i in slice_range:
                if i < img_data.shape[2]:
                    slice_data = img_data[:, :, i, 0]
                    slice_image = Image.fromarray((slice_data * 255).astype(np.uint8)).convert('RGB')
                    subject_slices.append(slice_image)
                    valid_slices += 1
            
            if valid_slices > 0:
                slices.extend(subject_slices)
                labels.extend([label_idx] * valid_slices)
                subject_ids.extend([subject_id] * valid_slices)
                #print(f"Subject {subject_id}: Processed {valid_slices} valid slices")
    
    return slices, labels, subject_ids

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_acc = 0.0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        epoch_train_loss = running_loss/len(train_loader)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        epoch_val_loss = val_loss/len(val_loader)
        
        # Step the scheduler based on validation accuracy
        scheduler.step(val_acc)
        
        # Store metrics for plotting
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {epoch_train_loss:.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.2f}%')
        print(f'Validation Accuracy: {val_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'New best model saved with validation accuracy: {val_acc:.2f}%')
        print('-' * 60)

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('loss_curves.png')
    plt.close()
    display(IPythonImage('loss_curves.png'))

    # Plot accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_accs, label='train')
    plt.plot(val_accs, label='validation')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('accuracy_curves.png')
    plt.close()
    display(IPythonImage('accuracy_curves.png'))
    plt.close()

    # Save metrics to file for later plotting
    np.save('training_metrics.npy', {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    })

    return best_val_acc

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot both normalized and raw count confusion matrices"""
    cm = confusion_matrix(y_true, y_pred)
    # Create a figure with 1 plot
    fig, (ax1) = plt.subplots(figsize=(10, 8))
    # Plot normalized confusion matrix (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1,
                annot_kws={"size": 16})
    ax1.set_ylabel('True Label', fontsize = 18)
    ax1.set_xlabel('Predicted Label', fontsize=18)
    ax1.set_title('Normalized Confusion Matrix', fontsize=20)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    # plt.show()
    display(IPythonImage('confusion_matrix.png'))

# Main training code
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    excel_path = 'oasis_cross-sectional-clean_MR1_only.xlsx'
    df = pd.read_excel(excel_path)
    
    # Combine CDR 1.0 and 2.0 into a single class
    df['CDR'] = df['CDR'].apply(lambda x: 1.0 if x == 2.0 else x)
    
    print("\nPreparing dataset from all discs...")
    all_slices, all_labels, all_subject_ids = prepare_data(df, transform=None)
    
    # Convert to numpy arrays for GroupKFold
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)
    
    outer_split = GroupKFold(n_splits=5)
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(outer_split.split(all_slices, all_labels, groups=all_subject_ids)):
        if fold_idx == 0:  
            # Get the test data
            test_slices = [all_slices[i] for i in test_idx]
            test_labels = all_labels[test_idx]
            test_subjects = all_subject_ids[test_idx]
            
            # Get the train+val data
            train_val_slices = [all_slices[i] for i in train_val_idx]
            train_val_labels = all_labels[train_val_idx]
            train_val_subjects = all_subject_ids[train_val_idx]
            
            # Now split train+val into train and validation
            inner_split = GroupKFold(n_splits=4) 
            for inner_fold_idx, (train_idx, val_idx) in enumerate(inner_split.split(train_val_slices, 
                                                                                   train_val_labels,
                                                                                   groups=train_val_subjects)):
                if inner_fold_idx == 0:  # Use first fold
                    # Get the final train data
                    train_slices = [train_val_slices[i] for i in train_idx]
                    train_labels = train_val_labels[train_idx]
                    train_subjects = train_val_subjects[train_idx]
                    
                    # Get the validation data
                    val_slices = [train_val_slices[i] for i in val_idx]
                    val_labels = train_val_labels[val_idx]
                    val_subjects = train_val_subjects[val_idx]
                    break
            break
    
    # Apply transforms
    train_slices_transformed = []
    val_slices_transformed = []
    test_slices_transformed = []
    
    for slice_img in train_slices:
        if train_transform:
            slice_img = train_transform(slice_img)
        train_slices_transformed.append(slice_img)
    
    for slice_img in val_slices:
        if val_transform:
            slice_img = val_transform(slice_img)
        val_slices_transformed.append(slice_img)
    
    for slice_img in test_slices:
        if test_transform:
            slice_img = test_transform(slice_img)
        test_slices_transformed.append(slice_img)
    
    # Convert labels to tensors
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    val_labels = torch.tensor(val_labels, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    # Create datasets
    train_dataset = MRIDataset(train_slices_transformed, train_labels)
    val_dataset = MRIDataset(val_slices_transformed, val_labels)
    test_dataset = MRIDataset(test_slices_transformed, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize model, criterion, optimizer with weight decay
    model = ResNeXt50(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.001,  
        weight_decay=0.01,  
        betas=(0.9, 0.999)
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    # Train the model with scheduler
    print("\nStarting training...")
    best_val_acc = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=7, device=device)
    print(f"\nBest validation accuracy achieved: {best_val_acc:.2f}%")
    
    # Load best model and evaluate on test set
    print("\nEvaluating on held-out test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # Test the model on the held-out test set
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays and save
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Save predictions and true labels
    np.save('predictions.npy', all_preds)
    np.save('true_labels.npy', all_labels)

    test_acc = 100. * test_correct / test_total
    print("\nFinal Results:")
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print('-' * 60)
    
    # Calculate metrics on test set
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    print("\nTest Set Class Distribution:")
    print("Unique classes in true labels:", np.unique(all_labels))
    print("Unique classes in predictions:", np.unique(all_preds))
    print("\nCDR Score Distribution in Test Set:")
    for idx in range(3):
        cdr_score = idx_to_cdr[idx]
        count = np.sum(all_labels == idx)
        print(f"CDR {cdr_score} ({cdr_label_map[cdr_score]}): {count} samples")
    
    # Get unique classes present in the data
    unique_classes = list(range(3))
    present_cdr_scores = [idx_to_cdr[cls] for cls in unique_classes]
    present_labels = [cdr_label_map[score] for score in present_cdr_scores]
    
    # F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'\nWeighted F1 Score: {f1:.4f}')
    

    plot_confusion_matrix(all_labels, all_preds, present_labels)