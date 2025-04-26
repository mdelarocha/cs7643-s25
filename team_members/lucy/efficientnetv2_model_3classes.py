import gcsfs
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from glob import glob
import math
import tempfile
import nibabel as nib
import logging
import shutil
import math
from sklearn.metrics import confusion_matrix, roc_curve, auc, f1_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from collections import Counter, defaultdict
# Suppress nibabel warnings about affine/origin issues
nib_log = logging.getLogger("nibabel")
nib_log.setLevel(logging.ERROR)  #non-critical warnings

# ----------------------------
# GCSFS: Load subject from GCS with multiple fallback methods

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

fs = gcsfs.GCSFileSystem(token='anon')
bucket_base = 'oasis-1-dataset-13635/oasis_raw'

def load_subject_data_from_gcs(subject_id, verbose=False):
    for disc_num in range(1, 13):
        t88_path = f"{bucket_base}/disc{disc_num}/{subject_id}/PROCESSED/MPRAGE/T88_111"

        n4_hdr = f"{t88_path}/{subject_id}_mpr_n4_anon_111_t88_masked_gfc.hdr"
        n4_img = f"{t88_path}/{subject_id}_mpr_n4_anon_111_t88_masked_gfc.img"
        n3_hdr = f"{t88_path}/{subject_id}_mpr_n3_anon_111_t88_masked_gfc.hdr"
        n3_img = f"{t88_path}/{subject_id}_mpr_n3_anon_111_t88_masked_gfc.img"

        try:
            if fs.exists(n4_hdr) and fs.exists(n4_img):
                hdr_file, img_file = n4_hdr, n4_img
            elif fs.exists(n3_hdr) and fs.exists(n3_img):
                hdr_file, img_file = n3_hdr, n3_img
            else:
                if verbose:
                    logger.info(f"Missing masked files for {subject_id} on disc {disc_num}")
                continue  # Try next disc

            # Create a temporary directory to work with the files
            temp_dir = tempfile.mkdtemp()
            try:
                # Step 1: Download the files to the temp directory with fixed names
                local_hdr = os.path.join(temp_dir, f"{subject_id}.hdr")
                local_img = os.path.join(temp_dir, f"{subject_id}.img")
                local_nii = os.path.join(temp_dir, f"{subject_id}.nii.gz")
                
                with fs.open(hdr_file, 'rb') as f:
                    with open(local_hdr, 'wb') as out:
                        out.write(f.read())
                with fs.open(img_file, 'rb') as f:
                    with open(local_img, 'wb') as out:
                        out.write(f.read())
                
                # Step 2: Try multiple methods to load the image
                data = None
                metadata = None
                
                # Method 1: Try nibabel with explicit convert to NIFTI
                try:
                    logger.info(f"Trying nibabel load for {subject_id}")
                    analyze_img = nib.load(local_hdr)
                    nib.save(analyze_img, local_nii)
                    img = sitk.ReadImage(local_nii)
                    data = sitk.GetArrayFromImage(img)
                    data = np.transpose(data, (2, 1, 0))  # Reorder to (Z, Y, X)
                    
                    metadata = {
                        'subject_id': subject_id,
                        'disc': disc_num,
                        'dimensions': data.shape,
                        'origin': img.GetOrigin(),
                        'spacing': img.GetSpacing(),
                        'direction': img.GetDirection()
                    }
                    logger.info(f"Successfully loaded {subject_id} with nibabel+sitk")
                except Exception as e:
                    logger.warning(f"Nibabel method failed for {subject_id}: {e}")
                
                # Method 2: Try direct nibabel with numpy conversion
                if data is None:
                    try:
                        logger.info(f"Trying direct nibabel for {subject_id}")
                        analyze_img = nib.load(local_hdr)
                        data = analyze_img.get_fdata()
                        # Note: Might need to adjust the orientation based on your requirements
                        
                        metadata = {
                            'subject_id': subject_id,
                            'disc': disc_num,
                            'dimensions': data.shape,
                            'origin': (0, 0, 0),  # Default values since we're not using sitk
                            'spacing': analyze_img.header.get_zooms(),
                            'direction': (1, 0, 0, 0, 1, 0, 0, 0, 1)  # Identity direction
                        }
                        logger.info(f"Successfully loaded {subject_id} with direct nibabel")
                    except Exception as e:
                        logger.warning(f"Direct nibabel method failed for {subject_id}: {e}")
                
                # Method 3: Try direct SimpleITK
                if data is None:
                    try:
                        logger.info(f"Trying direct SimpleITK for {subject_id}")
                        img = sitk.ReadImage(local_hdr)
                        data = sitk.GetArrayFromImage(img)
                        data = np.transpose(data, (2, 1, 0))
                        
                        metadata = {
                            'subject_id': subject_id,
                            'disc': disc_num,
                            'dimensions': data.shape,
                            'origin': img.GetOrigin(),
                            'spacing': img.GetSpacing(),
                            'direction': img.GetDirection()
                        }
                        logger.info(f"Successfully loaded {subject_id} with direct SimpleITK")
                    except Exception as e:
                        logger.warning(f"Direct SimpleITK method failed for {subject_id}: {e}")
                
                # If all methods failed
                if data is None:
                    logger.error(f"All loading methods failed for {subject_id} on disc {disc_num}")
                    continue
                
                return data, metadata
                
            finally:
                # Clean up the temporary directory
                shutil.rmtree(temp_dir)
                
        except Exception as e:
            logger.error(f"Error processing {subject_id} on disc {disc_num}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

    # If not found in any disc or all loading attempts failed
    logger.warning(f"Could not load {subject_id} from any disc")
    return None, None


def preprocess_scan(data):
    eps = 1e-6  # Tolerance for flat images or float precision
    if not np.isfinite(data).all():  # Checks for NaN and inf
        return np.full_like(data, 0.5)  # Use neutral value instead of zeros

    min_val = data.min()
    max_val = data.max()
    range_ = max_val - min_val

    if range_ < eps:
        return np.full_like(data, 0.5)  # Use neutral gray instead of zeros

    return (data - min_val) / (range_ + eps)



def save_processed_data(data, metadata, subject_id, output_dir):
    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    np.save(os.path.join(subject_dir, "brain_scan.npy"), data)
    with open(os.path.join(subject_dir, "metadata.txt"), "w") as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

def create_oasis_dataset_from_gcs(csv_path, output_dir, verbose=False):

    df = pd.read_excel(csv_path)
    df['processed'] = False
    os.makedirs(output_dir, exist_ok=True)
    
    def map_cdr_label (cdr):
        if cdr == 0.0:
            return "nondemented"
        elif cdr == 0.5:
            return "very mild dementia"
        elif cdr >= 1.0:
            return "mild to moderate dementia"
        else:
            return "unknown"  #fallback four unexpected values
    
    for index, row in tqdm(df.iterrows(), total=len(df)):
        subject_id = row['ID']
        cdr = row['CDR']
        if pd.isna(cdr): 
            continue
        cdr = float(cdr)
        class_label = map_cdr_label (cdr)

        if class_label is None: 
            continue #skip any unlabeled cdr

        output_path = os.path.join(output_dir, class_label)
        os.makedirs(output_path, exist_ok=True)

        data, metadata = load_subject_data_from_gcs(subject_id, verbose=verbose)
        if data is None: 
            continue

        data = preprocess_scan(data)
        save_processed_data(data, metadata, subject_id, output_path)
        df.at[index, 'processed'] = True

    df.to_excel(os.path.join(output_dir, "processing_status.xlsx"), index=False)
    return df

# ----------------------------
# Dataset class
# ----------------------------
class ProcessedOASISDataset(Dataset):
    def __init__(self, data_dir, df_path, transform=None, num_slices=5):
        self.data_dir = data_dir
        self.transform = transform
        self.num_slices = num_slices
        self.df = pd.read_excel(df_path)
        self.df['ID'] = self.df['ID'].astype(str).str.strip().str.upper()
        if 'processed' in self.df.columns:
            self.df = self.df[self.df['processed'] == True].reset_index(drop=True)
        self.subject_paths = []

        for class_dir in sorted(glob(os.path.join(data_dir, "*"))):
            if not os.path.isdir(class_dir): continue
            class_label = os.path.basename(class_dir)
            for subject_dir in sorted(glob(os.path.join(class_dir, "*"))):
                if not os.path.isdir(subject_dir): continue
                subject_id = os.path.basename(subject_dir)
                brain_scan_path = os.path.join(subject_dir, "brain_scan.npy")
                if os.path.exists(brain_scan_path):
                    self.subject_paths.append({'path': brain_scan_path, 'subject_id': subject_id, 'class': class_label})

        df_ids = self.df['ID'].values
        self.subject_paths = [s for s in self.subject_paths if s['subject_id'].strip().upper() in df_ids]

        # Add debugging code to find missing subjects
        csv_ids = set(self.df['ID'].values)
        found_ids = set([s['subject_id'].strip().upper() for s in self.subject_paths])
        missing_ids = csv_ids - found_ids
        print(f"Total subjects in CSV: {len(csv_ids)}")
        print(f"Total subjects found with images: {len(found_ids)}")
        print(f"Missing subjects: {missing_ids}")

    def __len__(self):
        return len(self.subject_paths)

    def __getitem__(self, idx):
        subject = self.subject_paths[idx]
        subject_id = subject['subject_id'] 
        class_label = subject['class']

        brain_scan = np.load(subject['path'])

        # Crop to range [18:158] to avoid known noisy extremes
        #Includes the entire brain without noise
        #volume with 140 slices
        mri_volume = brain_scan[:, :, 18:158]  # shape: (H, W, 140) 

        # Compute variance per slice
        slice_variances = np.var(mri_volume, axis=(0, 1)) #how much pixel intensities differ within the slice
        slice_means = np.mean(mri_volume, axis=(0, 1)) #overall brightness of the slice

        # Step 1: Exclude very dark slices (likely near-empty)
        valid_slice_mask = slice_means > 0.01  # adjust threshold if needed
        valid_indices = np.where(valid_slice_mask)[0]

        # Step 2: Limit to mid-brain range 
        
        mid_range_mask = (valid_indices >= 20) & (valid_indices <= 120) #when using anatomical weighting Test Accuracy: 58.33% #Test Macro F1 Score: 0.4862
        #restricts selection to only the specified range (110 - 30 + 1 = 81 slices)
        #mid_range_mask = (valid_indices >= 30) & (valid_indices <= 110) #Test Accuracy: 56.76% Test Macro F1 Score: 0.3798 (using 7 slices)
        #mid_range_mask = (valid_indices >= 32) & (valid_indices <= 115) #Test Accuracy: 40.54%
        #mid_range_mask = (valid_indices >= 40) & (valid_indices <= 100) #Test Accuracy: 16.22%
       
        filtered_indices = valid_indices[mid_range_mask]
        num_slices = self.num_slices 


        # If there are fewer than num_slices valid slices, fallback to the full valid range
        if len(filtered_indices) < num_slices:
            # Fallback: choose from full valid range
            fallback_indices = np.argsort(slice_variances[valid_indices])[-num_slices:]
            selected_indices = valid_indices[fallback_indices]
        else:
            # Step 3: Apply anatomical weighting
            # Use a Gaussian centered at index 70 (approximate brain center in [18:158] range)
            sigma = 10  # Controls how wide the weighting is; 10–15 is typical
            anatomical_weights = np.exp(-0.5 * ((filtered_indices - 70) / sigma) ** 2) # Gaussian function for weight

            # Combine variance and anatomical weight — prioritize informative & central slices
            weighted_variances = slice_variances[filtered_indices] * anatomical_weights # Combine variance and weight
            top_k = np.argsort(weighted_variances)[-num_slices:]
            selected_indices = filtered_indices[top_k]     

        #Without anatomical weighting
         #num_slices = self.num_slices 
        # if len(filtered_indices) < num_slices:
        #     #Fallback: choose from full valid range
        #     fallback_indices = np.argsort(slice_variances[valid_indices])[-num_slices:]
        #     selected_indices = valid_indices[fallback_indices]
        # else:
        #     filtered_variances = slice_variances[filtered_indices]
        #     top_k = np.argsort(filtered_variances)[-num_slices:]
        #     selected_indices = filtered_indices[top_k]

        #Ensure consistent ordering
        selected_indices = sorted(selected_indices)

        #Extract selected slices and prepare tensor
        slices = mri_volume[:, :, selected_indices] #Shape (H,W,num_slices)
        slices = np.transpose(slices, (2, 0, 1))  #Shape (num_slices,H,W)
        mri_slice = torch.tensor(slices, dtype=torch.float32)


        if self.transform:
            mri_slice = self.transform(mri_slice)
        subject_row = self.df[self.df['ID'] == subject_id].iloc[0]
        
        # New code with NaN handling:
        clinical_data_raw = [
            1.0 if subject_row['M/F'] == 'M' else 0.0,
            1.0 if subject_row['Hand'] == 'R' else 0.0,
            float(subject_row['Age']) if not pd.isna(subject_row['Age']) else 0.0,
            float(subject_row['Educ']) if not pd.isna(subject_row['Educ']) else 0.0,
            float(subject_row['SES']) if not pd.isna(subject_row['SES']) else 0.0,
            float(subject_row['MMSE']) if not pd.isna(subject_row['MMSE']) else 0.0,
        ]
        
        # Normalize safely
        clinical_data = [
            clinical_data_raw[0],
            clinical_data_raw[1],
            (clinical_data_raw[2] - 70.0) / 20.0 if clinical_data_raw[2] != 0.0 else 0.0,
            (clinical_data_raw[3] - 3.0) / 2.0 if clinical_data_raw[3] != 0.0 else 0.0,
            (clinical_data_raw[4] - 2.5) / 2.5 if clinical_data_raw[4] != 0.0 else 0.0,
            (clinical_data_raw[5] - 25.0) / 5.0 if clinical_data_raw[5] != 0.0 else 0.0,
        ]
        
        clinical_data = torch.tensor(clinical_data, dtype=torch.float32)
        
        # NaN handling:
        volume_data_raw = [
            float(subject_row['eTIV']) if not pd.isna(subject_row['eTIV']) else 0.0,
            float(subject_row['ASF']) if not pd.isna(subject_row['ASF']) else 0.0,
            float(subject_row['nWBV']) if not pd.isna(subject_row['nWBV']) else 0.0,
        ]
        
        volume_data = [
            (volume_data_raw[0] - 1400) / 200 if volume_data_raw[0] != 0.0 else 0.0,
            volume_data_raw[1],
            volume_data_raw[2],
        ]
        
        volume_data = torch.tensor(volume_data, dtype=torch.float32)
        
        # Add verification 
        assert not torch.isnan(clinical_data).any(), f"NaN in clinical_data for subject {subject_id}"
        assert not torch.isnan(volume_data).any(), f"NaN in volume_data for subject {subject_id}"
        
        # Label conversion
        label_to_index = {"nondemented": 0, "very mild dementia": 1, "mild dementia": 2, "moderate dementia": 2}
        label_index = label_to_index[class_label.lower()]
        
        return mri_slice, clinical_data, volume_data, label_index
    
# ----------------------------
#EfficientNetLite
# ----------------------------
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, groups=1):
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() #Applies the Sigmoid Linear Unit (SiLU) 
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride, survival_prob=None):
        super().__init__()
        exp_channels = int(in_channels * expansion_ratio)
        self.conv1 = ConvBnAct(in_channels, exp_channels, 3, stride)
        self.conv2 = nn.Sequential(nn.Conv2d(exp_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.has_skip = (stride == 1 and in_channels == out_channels)
        self.survival_prob = survival_prob
    def stochastic_depth(self, x): return x
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.has_skip: x = x + identity
        return x

class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_ratio, stride, kernel_size=3, survival_prob=None):
        super().__init__()
        exp_channels = int(in_channels * expansion_ratio)
        self.expand = ConvBnAct(in_channels, exp_channels, 1) if expansion_ratio != 1 else nn.Identity()
        self.dw = ConvBnAct(exp_channels, exp_channels, kernel_size, stride, groups=exp_channels)
        self.project = nn.Sequential(nn.Conv2d(exp_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.has_skip = (stride == 1 and in_channels == out_channels)
        self.survival_prob = survival_prob
    def stochastic_depth(self, x): return x
    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.dw(x)
        x = self.project(x)
        if self.has_skip: x = x + identity
        return x

class EfficientNetV2Lite(nn.Module):
    def __init__(self, model_config, num_classes=1000, in_channels = 5):
        super().__init__()
        self.features = [ConvBnAct(in_channels, model_config[0]['out_channels'], 3, 2)]
        block_idx = 0
        total_blocks = sum(c['num_layers'] for c in model_config)
        for stage_idx, cfg in enumerate(model_config):
            for i in range(cfg['num_layers']):
                stride = cfg['stride'] if i == 0 else 1
                in_ch = cfg['out_channels'] if i > 0 else (
                    model_config[stage_idx - 1]['out_channels'] if stage_idx > 0 else model_config[0]['out_channels'])
                block_cls = FusedMBConv if cfg['block_type'] == 'fused' else MBConv
                self.features.append(block_cls(in_ch, cfg['out_channels'], cfg['expansion_ratio'], stride))
        final_channels = model_config[-1]['out_channels']
        self.features.append(ConvBnAct(final_channels, final_channels * 2, 1))
        self.features = nn.Sequential(*self.features)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.out_dim = final_channels * 2
    def forward(self, x): return self.head(self.features(x))

def efficientnetv2_lite_s():
    return EfficientNetV2Lite([
        {'block_type': 'fused', 'out_channels': 24, 'expansion_ratio': 1, 'stride': 1, 'num_layers': 2},
        {'block_type': 'fused', 'out_channels': 48, 'expansion_ratio': 4, 'stride': 2, 'num_layers': 4},
        {'block_type': 'fused', 'out_channels': 64, 'expansion_ratio': 4, 'stride': 2, 'num_layers': 4},
        {'block_type': 'mbconv', 'out_channels': 128, 'expansion_ratio': 4, 'stride': 2, 'num_layers': 4},
        {'block_type': 'mbconv', 'out_channels': 160, 'expansion_ratio': 6, 'stride': 1, 'num_layers': 6},
        {'block_type': 'mbconv', 'out_channels': 256, 'expansion_ratio': 6, 'stride': 2, 'num_layers': 8},
    ])

# lighter version of EfficientNetV2 for better generalization on small dataset
simplified_config = [
    {'block_type': 'fused', 'out_channels': 16, 'expansion_ratio': 1, 'stride': 1, 'num_layers': 1},
    {'block_type': 'fused', 'out_channels': 32, 'expansion_ratio': 2, 'stride': 2, 'num_layers': 2},
    {'block_type': 'mbconv', 'out_channels': 64, 'expansion_ratio': 2, 'stride': 2, 'num_layers': 2},
]

def efficientnetv2_lite_simplified(in_channels=5):
    return EfficientNetV2Lite(simplified_config, in_channels=in_channels)


class EfficientMultiModalNet(nn.Module):
    def __init__(self, num_classes=3, num_slices=5):
        super().__init__()
        #self.efficientnet = efficientnetv2_lite_s()
        self.efficientnet = efficientnetv2_lite_simplified(in_channels=num_slices)
        self.image_feature_dim = self.efficientnet.out_dim
        #I added batch norm to stabilize model and reduced dropout from 0.3 to 0.2
        self.clinical_fc = nn.Sequential(nn.Linear(6, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2)) 
        self.volume_fc = nn.Sequential(nn.Linear(3, 16), nn.BatchNorm1d(16),nn.ReLU(), nn.Dropout(0.2))
        #I added batch norm and lowered dropout from 0.5 to 0.2
        self.classifier = nn.Sequential(nn.Linear(self.image_feature_dim + 32 + 16, 128), nn.BatchNorm1d(128),nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_classes)) 
        
        # Initialize bias of the final linear layer
        final_fc = self.classifier[-1]
        if isinstance(final_fc, nn.Linear):
            nn.init.constant_(final_fc.bias, 0.01)
    
    def forward(self, images, clinical_data, volume_data):
        img_feat = self.efficientnet(images)
        print(f"Image feature norm: {torch.norm(img_feat).item()}")

        clin_feat = self.clinical_fc(clinical_data)
        print(f"Clinical feature norm: {torch.norm(clin_feat).item()}")

        vol_feat = self.volume_fc(volume_data)
        print(f"Volume feature norm: {torch.norm(vol_feat).item()}")

        x = self.classifier(torch.cat([img_feat, clin_feat, vol_feat], dim=1))
        x = torch.nan_to_num(x)  # Clamp NaNs/infs to 0
        return x
    
# ----------------------------
# EfficientNetLite End
# ----------------------------

#----------------------------
# HybridEfficient start
# ----------------------------
class HybridEfficientNet(nn.Module):
    def __init__(self, num_classes=3, num_slices=5):
        super().__init__()

        # Use EfficientNet as the image feature extractor
        self.efficientnet = efficientnetv2_lite_simplified(in_channels=num_slices)
        self.image_feature_dim = self.efficientnet.out_dim

        # Clinical data path (from SimpleModel)
        self.fc_clinical = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Volume data path (from SimpleModel)
        self.fc_volume = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Classifier head (from SimpleModel)
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feature_dim + 16 + 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images, clinical_data, volume_data):
        images = torch.nan_to_num(images)
        clinical_data = torch.nan_to_num(clinical_data)
        volume_data = torch.nan_to_num(volume_data)

        img_feat = self.efficientnet(images)
        clin_feat = self.fc_clinical(clinical_data)
        vol_feat = self.fc_volume(volume_data)

        x = torch.cat([img_feat, clin_feat, vol_feat], dim=1)
        return self.classifier(x)
#----------------------------
# HybridEfficient End
# ----------------------------


# ----------------------------
# Simple CNN Model Start
# ----------------------------
#I used this model as a baseline to get better results since my efficientnet model had poor results
class SimpleModel(nn.Module):
    def __init__(self, num_classes=3, in_channels=5):
        super().__init__()
        # Image feature extraction path
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Clinical data path with batch norm and dropout
        self.fc_clinical = nn.Sequential(
            nn.Linear(6, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Volume data path with batch norm and dropout
        self.fc_volume = nn.Sequential(
            nn.Linear(3, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier with batch norm and multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(1568 + 16 + 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, images, clinical_data, volume_data):
        # Safety measures for handling NaNs
        images = torch.nan_to_num(images)
        clinical_data = torch.nan_to_num(clinical_data)
        volume_data = torch.nan_to_num(volume_data)
        
        # CNN feature extraction
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        
        # Process clinical and volume data
        clinical_feat = self.fc_clinical(clinical_data)
        volume_feat = self.fc_volume(volume_data)
        
        # Combine all features and classify
        combined = torch.cat([x, clinical_feat, volume_feat], dim=1)
        output = self.classifier(combined)
        
        return output
# ----------------------------
# Simple CNN Model End
# ----------------------------

# Visualization Functions
def plot_training_curves(metrics):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_accs'], label='Train Accuracy')
    plt.plot(metrics['val_accs'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()

# def plot_confusion_matrix(y_true, y_pred, class_names):
#     """Plot confusion matrix with percentages"""
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(10, 8))
#     cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
#                 xticklabels=class_names, yticklabels=class_names)
#     plt.ylabel('True Label')
#     plt.xlabel('Predicted Label')
#     plt.title('Normalized Confusion Matrix')
#     plt.tight_layout()
#     plt.savefig('confusion_matrix.png')
#     plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot both normalized and raw count confusion matrices"""
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot normalized confusion matrix (percentages)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title('Normalized Confusion Matrix')
    
    # Plot raw count confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    ax2.set_title('Count Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def plot_sensitivity_specificity(metrics, class_names):
    """Plot sensitivity and specificity trends per class"""
    plt.figure(figsize=(15, 10))
    
    # Convert list of arrays to 2D array for easier plotting
    sensitivities = np.array(metrics['class_sensitivities'])
    specificities = np.array(metrics['class_specificities'])
    
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 2, i+1)
        plt.plot(sensitivities[:, i], label='Sensitivity')
        plt.plot(specificities[:, i], label='Specificity')
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        plt.title(f'Class: {class_name}')
        plt.legend()
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sensitivity_specificity.png')
    plt.show()

def plot_roc_curves(model, test_loader, class_names):
    """Plot ROC curves for each class"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for mri, clinical, volume, labels in test_loader:
            mri, clinical, volume = mri.to(device), clinical.to(device), volume.to(device)
            outputs = model(mri, clinical, volume)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Binarize labels for ROC calculation
    n_classes = len(class_names)
    all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['blue', 'red', 'green', 'orange'])
    
    for i, color, class_name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig('roc_curves.png')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def visualize_brain_slices(dataset, num_subjects=2, num_slices_to_show=2):
    """
    For each class in `dataset`, pick up to `num_subjects` subjects and display
    `num_slices_to_show` evenly spaced slices per subject.
    """
    # 1) Group indices by whatever label your dataset returns
    class_indices = defaultdict(list)
    for idx in range(len(dataset)):
        *_, label = dataset[idx]
        if len(class_indices[label]) < num_subjects:
            class_indices[label].append(idx)
        # stop once every discovered class has enough examples
        if all(len(v) >= num_subjects for v in class_indices.values()):
            break

    # 2) Figure out which slice‐channels to show
    #    (we just pick from the first class we found)
    first_label = next(iter(class_indices))
    example, *_, _ = dataset[class_indices[first_label][0]]
    total_slices  = example.shape[0]
    slice_indices = np.linspace(
        0, total_slices - 1, num_slices_to_show, dtype=int
    )

    # 3) Set up figure grid
    n_classes = len(class_indices)
    n_cols    = num_subjects * num_slices_to_show
    fig, axes = plt.subplots(
        n_classes, n_cols,
        figsize=(2 * n_cols, 2 * n_classes),
        squeeze=False
    )

    # 4) Fill in each subplot
    for row, label in enumerate(class_indices):
        for subj_pos, idx in enumerate(class_indices[label]):
            volume, *_, _ = dataset[idx]   # volume.shape == (num_slices, H, W)
            for slice_pos, sl in enumerate(slice_indices):
                ax = axes[row, subj_pos * num_slices_to_show + slice_pos]
                ax.imshow(volume[sl].numpy(), cmap='gray')
                ax.axis('off')
                if row == 0:
                    ax.set_title(f"slice {sl}", fontsize=8)
                if slice_pos == 0:
                    ax.set_ylabel(str(label), fontsize=10)

    plt.tight_layout()
    plt.show()



def visualize_3d_brain(dataset, index=0, slice_step=20):
    subject_path = dataset.subject_paths[index]['path']
    brain_scan = np.load(subject_path)

    class_label = dataset.subject_paths[index]['class']

    depth, height, width = brain_scan.shape

    # Get evenly spaced slice indices for each plane
    axial_indices = list(range(10, depth - 10, slice_step))
    coronal_indices = list(range(10, height - 10, slice_step))
    sagittal_indices = list(range(10, width - 10, slice_step))

    # Plot setup
    fig = plt.figure(figsize=(20, 12))  # Wider for more space

    # Axial View
    for i, z in enumerate(axial_indices):
        ax = fig.add_subplot(3, len(axial_indices), i + 1)
        ax.imshow(brain_scan[z, :, :], cmap='gray')
        ax.set_title(f'Axial z={z}', fontsize=8)
        ax.axis('off')

    # Coronal View
    for i, y in enumerate(coronal_indices):
        ax = fig.add_subplot(3, len(coronal_indices), len(axial_indices) + i + 1)
        ax.imshow(brain_scan[:, y, :], cmap='gray')
        ax.set_title(f'Coronal y={y}', fontsize=8)
        ax.axis('off')

    # Sagittal View
    for i, x in enumerate(sagittal_indices):
        ax = fig.add_subplot(3, len(sagittal_indices), 2 * len(axial_indices) + i + 1)
        ax.imshow(brain_scan[:, :, x], cmap='gray')
        ax.set_title(f'Sagittal x={x}', fontsize=8)
        ax.axis('off')

    plt.suptitle(f'Brain Visualization: {class_label}', fontsize=16, y=0.92)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space for title
    plt.savefig('3d_brain_visualization_fixed.png')
    plt.show()


def plot_saliency_maps(model, dataset, indices, class_names):
    """Generate saliency maps to highlight important regions for classification"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    fig, axes = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)), constrained_layout=True)

    
    for i, idx in enumerate(indices):
        # Get data
        mri, clinical, volume, label = dataset[idx]
        mri = mri.unsqueeze(0).to(device)
        clinical = clinical.unsqueeze(0).to(device)
        volume = volume.unsqueeze(0).to(device)
        
        # Register hook for gradients
        mri.requires_grad_()
        
        # Forward pass
        output = model(mri, clinical, volume)
        
        # Get predicted class
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        
        # Backprop to get gradient
        model.zero_grad()
        output[0, predicted_class].backward()
        
        # Get gradients
        gradients = mri.grad.abs().cpu().numpy()[0, 0]
        
        # Normalize gradients for better visualization
        gradients = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
        
        # Original image
        axes[i, 0].imshow(mri.detach().cpu().numpy()[0, 0], cmap='gray')
        axes[i, 0].set_title(f'Original:\n {class_names[label]}',fontdict={'fontsize': 10})
        axes[i, 0].axis('off')
        
        # Saliency map
        axes[i, 1].imshow(gradients, cmap='hot')
        axes[i, 1].set_title(f'Saliency:\n {class_names[predicted_class]}', fontdict={'fontsize': 10})
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(mri.detach().cpu().numpy()[0, 0], cmap='gray')
        axes[i, 2].imshow(gradients, cmap='hot', alpha=0.5)
        axes[i, 2].set_title('Overlay', fontdict={'fontsize': 10})
        axes[i, 2].axis('off')
    
    plt.savefig('saliency_maps.png', dpi=200)
    plt.show()

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # class-wise weights (torch.tensor of shape [num_classes])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')  # no alpha yet
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha[targets]  # select per-sample alpha weight
            ce_loss = at * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# Function to set up the model, optimizer, criterion
# Initializes the model, loss function, optimizer, scheduler, and scaler.
# This function is called inside `train_model()` to set up all training components.
#prepares everything needed for training — but it does NOT run training itself.

def setup_training(model, num_epochs=10, lr=1e-4, weight_decay=1e-5, gamma=2.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    # # Sample weights based on class counts in sampled training set
    # class_counts = torch.tensor([49, 34, 44, 36], dtype=torch.float) 
    # weights = 1.0 / class_counts
    # weights = weights / weights.sum()
    
    # Loss function
    criterion = FocalLoss(gamma=gamma)  

    # AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()
    
    return model, criterion, optimizer, scheduler, scaler, device



# Function to train for one epoch
def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0

    for mri, clinical, volume, labels in tqdm(train_loader, desc="Training"):
        try:
            # Force all inputs to float32
            mri = mri.to(device, dtype=torch.float32)
            clinical = clinical.to(device, dtype=torch.float32)
            volume = volume.to(device, dtype=torch.float32)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Debug input data
            if torch.isnan(mri).any():
                print("NaNs in MRI input data, skipping batch")
                continue

            # Disable AMP: use standard training loop
            # if scaler:
            #     with torch.amp.autocast('cuda'):
            #         outputs = model(mri, clinical, volume)
            #         if train_total == 0:
            #             print("Logits (first batch):", outputs[:2])
            #         loss = criterion(outputs, labels)

            #     if torch.isnan(outputs).any() or torch.isnan(loss):
            #         print("NaN encountered in loss or outputs! Skipping batch")
            #         continue

            #     scaler.scale(loss).backward()
            #     scaler.unscale_(optimizer)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            outputs = model(mri, clinical, volume)
            if train_total == 0:
                print("Logits (first batch):", outputs[:2])
            loss = criterion(outputs, labels)

            if torch.isnan(outputs).any() or torch.isnan(loss):
                print("NaN encountered in loss or outputs! Skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_correct += preds.eq(labels).sum().item()
            train_total += labels.size(0)

        except Exception as e:
            print(f"Error in training batch: {e}")
            continue

    # Calculate epoch metrics
    if train_total == 0:
        print("Warning: No valid training samples this epoch.")
        train_acc = 0.0
        train_loss = 0.0
    else:
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

    return train_loss, train_acc

    
    # Calculate epoch metrics
    if train_total == 0:
        print("Warning: No valid training samples this epoch.")
        train_acc = 0.0
        train_loss = 0.0
    else:
        train_acc = 100.0 * train_correct / train_total
        train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
    
    return train_loss, train_acc

# Function to validate the model
def validate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for mri, clinical, volume, labels in tqdm(val_loader, desc="Validation"):
            try:
                mri, clinical, volume, labels = mri.to(device), clinical.to(device), volume.to(device), labels.to(device)
                outputs = model(mri, clinical, volume)
                
                if torch.isnan(outputs).any():
                    print("NaN in validation outputs! Skipping batch.")
                    continue
                
                loss = criterion(outputs, labels)
                if torch.isnan(loss):
                    print("NaN in validation loss! Skipping batch.")
                    continue
                
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_correct += preds.eq(labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue
    
    # Calculate metrics
    if val_total == 0:
        print("Warning: No valid validation samples.")
        val_acc = 0.0
        val_loss = 0.0
    else:
        val_acc = 100.0 * val_correct / val_total
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    
    return val_loss, val_acc, all_preds, all_labels

# Function to calculate class-specific metrics
def calculate_metrics(all_labels, all_preds):
    try:
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        print(f"Confusion Matrix:\n{cm}")
        
        # Compute metrics per class
        TP = np.diag(cm)
        FN = cm.sum(axis=1) - TP
        FP = cm.sum(axis=0) - TP
        TN = cm.sum() - (TP + FP + FN)
        
        sensitivity = TP / (TP + FN + 1e-6)
        specificity = TN / (TN + FP + 1e-6)
        
        return cm, sensitivity, specificity
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None, None, None


# Main training loop function  
def train_model(model, train_loader, val_loader, num_epochs=30, lr=1e-6, weight_decay=1e-3, gamma=1.0):
    # Setup training components
    model, criterion, optimizer, scheduler, scaler, device = setup_training(
        model, num_epochs=num_epochs, lr=lr, weight_decay=weight_decay, gamma=gamma
    )

    # Initialize tracking metrics
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    class_sensitivities, class_specificities = [], []
    best_val_acc = 0.0

    # Main training loop
    for epoch in range(num_epochs):
        # Train one epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )

        # Validate model
        val_loss, val_acc, all_preds, all_labels = validate_model(
            model, val_loader, criterion, device
        )

        # Update learning rate scheduler **after** optimizer step
        scheduler.step()

        # Calculate and store metrics
        cm, sensitivity, specificity = calculate_metrics(all_labels, all_preds)

        # Store metrics for visualization
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if sensitivity is not None and specificity is not None:
            class_sensitivities.append(sensitivity)
            class_specificities.append(specificity)

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if sensitivity is not None:
            for i in range(len(sensitivity)):
                print(f"Class {i}: Sensitivity = {sensitivity[i]:.2%}, Specificity = {specificity[i]:.2%}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_oasis_model.pth")
            print("Saved best model.")
        print("-" * 60)

    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'class_sensitivities': class_sensitivities,
        'class_specificities': class_specificities
    }


# Function to evaluate model on test set
def evaluate_model(model, test_loader, class_names):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for mri, clinical, volume, labels in tqdm(test_loader, desc="Testing"):
            try:
                mri, clinical, volume, labels = mri.to(device), clinical.to(device), volume.to(device), labels.to(device)
                outputs = model(mri, clinical, volume)
                
                if torch.isnan(outputs).any():
                    continue
                    
                _, preds = outputs.max(1)
                test_correct += preds.eq(labels).sum().item()
                test_total += labels.size(0)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                print(f"Error in test batch: {e}")
                continue
    
    # Calculate accuracy
    test_acc = 100.0 * test_correct / test_total if test_total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Calculate F1 Score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Test Macro F1 Score: {macro_f1:.4f}")
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_preds, class_names)
    
    return test_acc, macro_f1, all_preds, all_labels


# Function for Sample Weights
def get_sample_weights(dataset, train_dataset):
    y_train_indices = train_dataset.indices
    y_train = [dataset.subject_paths[i]['class'] for i in y_train_indices]

    label_to_index = {
        "nondemented": 0,
        "very mild dementia": 1,
        "mild dementia": 2,
        "moderate dementia": 2
    }
    y_train_numeric = [label_to_index[label] for label in y_train]

    class_sample_counts = np.bincount(y_train_numeric)
    weights = 1.0 / class_sample_counts
    sample_weights = [weights[label] for label in y_train_numeric]

    return torch.DoubleTensor(sample_weights)


# ----------------------------
# MAIN EXECUTION
# ----------------------------

if __name__ == "__main__":
    script_dir = os.getcwd()
    data_dir = os.path.join(script_dir, "data")
    csv_path = os.path.join(data_dir, "oasis_cross-sectional-clean_MR1_only.xlsx")
    output_dir = os.path.join(script_dir, "processed_oasis_data")

    # Check if data is already processed
    if not os.path.exists(os.path.join(output_dir, "processing_status.xlsx")):
        result_df = create_oasis_dataset_from_gcs(csv_path=csv_path, output_dir=output_dir,  verbose=True)
        print(f"Successfully processed {result_df['processed'].sum()} out of {len(result_df)} subjects")
    else:
        print("Using previously processed data")

    # Enhance transform with more augmentations
    #Source data adapted from: https://pytorch.org/vision/main/transforms.html
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.45], [0.225])
    ])

    num_slices = 7 #adjust this one
    # Create dataset
    dataset = ProcessedOASISDataset(

        data_dir=output_dir, 
        df_path=os.path.join(output_dir, "processing_status.xlsx"), 
        transform=transform,
        num_slices=num_slices,
    )
    print(f"Dataset length: {len(dataset)}")

    # Debug dataset
    for i in range(min(5, len(dataset))):
        mri, clinical, volume, label = dataset[i]
        print(f"Sample {i}:")
        print(f"MRI shape: {mri.shape}, range: [{mri.min():.4f}, {mri.max():.4f}]")
        print(f"Clinical data: {clinical}, range: [{clinical.min():.4f}, {clinical.max():.4f}]")
        print(f"Volume data: {volume}, range: [{volume.min():.4f}, {volume.max():.4f}]")
        print(f"Label: {label}")
        print("-" * 50)
    
    # Visualize sample brain slices
    visualize_brain_slices(dataset)

    # Extract all unique subject IDs and shuffle
    all_subject_ids = list(set([s['subject_id'] for s in dataset.subject_paths]))
    np.random.seed(42)
    np.random.shuffle(all_subject_ids)

    # Helper: Map subject_id → class
    subject_id_to_class = {s['subject_id']: s['class'] for s in dataset.subject_paths}

    # # Find all moderate dementia subjects
    # moderate_subjects = [sid for sid, label in subject_id_to_class.items() if label == 'moderate dementia']

    # Split
    train_ratio, val_ratio = 0.7, 0.15
    n_subjects = len(all_subject_ids)
    train_size = int(n_subjects * train_ratio)
    val_size = int(n_subjects * val_ratio)

    train_subjects = all_subject_ids[:train_size]
    val_subjects = all_subject_ids[train_size:train_size + val_size]
    test_subjects = all_subject_ids[train_size + val_size:]

    # # Force 1 moderate dementia subject into each split if possible
    # mod_subjects = [s for s in all_subject_ids if subject_id_to_class[s] == "moderate dementia"]
    # if mod_subjects:
    #     # REMOVE moderate dementia subjects from any existing splits
    #     train_subjects = [s for s in train_subjects if s not in mod_subjects]
    #     val_subjects = [s for s in val_subjects if s not in mod_subjects]
    #     test_subjects = [s for s in test_subjects if s not in mod_subjects]

    #     # Then ADD one to each split
    #     train_subjects.append(mod_subjects[0])
    #     val_subjects.append(mod_subjects[1 % len(mod_subjects)])
    #     test_subjects.append(mod_subjects[2 % len(mod_subjects)])

    # Create subject-based indices for each split
    train_indices = [i for i in range(len(dataset)) if dataset.subject_paths[i]['subject_id'] in train_subjects]
    val_indices = [i for i in range(len(dataset)) if dataset.subject_paths[i]['subject_id'] in val_subjects]
    test_indices = [i for i in range(len(dataset)) if dataset.subject_paths[i]['subject_id'] in test_subjects]

    # Log class distribution
    train_classes = [dataset.subject_paths[i]['class'] for i in train_indices]
    val_classes = [dataset.subject_paths[i]['class'] for i in val_indices]
    test_classes = [dataset.subject_paths[i]['class'] for i in test_indices]

    print("Training set class distribution:", Counter(train_classes))
    print("Validation set class distribution:", Counter(val_classes))
    print("Test set class distribution:", Counter(test_classes))

    # Create subset datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    print(f"Split complete: {len(train_dataset)} train, {len(val_dataset)} validation, {len(test_dataset)} test samples")
    print(f"Using {len(train_subjects)} training subjects, {len(val_subjects)} validation subjects, {len(test_subjects)} test subjects")
    
     #Create data loaders with balanced sampling for training
    sample_weights = get_sample_weights(dataset, train_dataset)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Define data loaders hyperparameters
    train_batch_size = 32
    test_batch_size = 32
  
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size ,  sampler=train_sampler, num_workers=0) # Use the sampler
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)
    
    # Define class names for visualization``
    class_names = ["nondemented", "very mild dementia", "mild to moderate dementia"]

    # Define training hyperparameters
    #Adjust hyperparameters here

    lr = 1e-4 #1e-5
    weight_decay = 1e-4 #1e-3
    num_epochs = 30
    gamma= 1.0  #focuses on more difficult classes
    

    # Create and train model
    model = HybridEfficientNet(num_classes=3, num_slices=num_slices)
    #model = EfficientMultiModalNet(num_classes=3)
    #model = SimpleModel(num_classes=3)

    #this is not needed for the HybridEfficientNet model
    #model.apply(lambda m: nn.init.xavier_normal_(m.weight) if isinstance(m, nn.Linear) else None)

    
    # Monitor class distribution in training data
    class_samples_seen = {0: 0, 1: 0, 2: 0}
    print("Analyzing class distribution in one epoch with weighted sampling:")
    for batch_idx, (_, _, _, labels) in enumerate(train_loader):
        for label in labels:
            class_samples_seen[label.item()] += 1
    print(f"Samples seen per class in one epoch: {class_samples_seen}")
    print(f"Total samples: {sum(class_samples_seen.values())}")

    # Count all .npy files recursively (including in subfolders)
    total_augmented_samples = len(glob(os.path.join(output_dir, '**', '*.npy'), recursive=True))
    print(f"Total augmented + original samples in training directory: {total_augmented_samples}")
    
    
    model, metrics = train_model(model, train_loader, val_loader, num_epochs=num_epochs, lr=lr, weight_decay= weight_decay, gamma=gamma) #I tried various learning rates and weight decays
    
    # Visualize training results
    plot_training_curves(metrics)
    plot_sensitivity_specificity(metrics, class_names)
    
    # Evaluate on test set
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    test_correct, test_total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for mri, clinical, volume, labels in tqdm(test_loader, desc="Testing"):
            mri, clinical, volume, labels = mri.to(device), clinical.to(device), volume.to(device), labels.to(device)
            
            outputs = model(mri, clinical, volume)
            _, preds = outputs.max(1)
            
            test_correct += preds.eq(labels).sum().item()
            test_total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = 100.0 * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")

    #Evaluate with Macro-Averaged F1 Score
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Test Macro F1 Score: {macro_f1:.4f}")

    
    # Plot confusion matrix and ROC curves
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_roc_curves(model, test_loader, class_names)
    
    # Generate some saliency maps
    sample_indices = [i for i in range(min(8, len(test_dataset)))]
    plot_saliency_maps(model, test_dataset, sample_indices, class_names)
    
    # Visualize a 3D brain
    visualize_3d_brain(dataset, index=0)
    
    print("All visualizations have been saved to the current directory.")




