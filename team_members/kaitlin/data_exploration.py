import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import filters, measure
from scipy.ndimage import gaussian_filter
import gcsfs

def calculate_slice_metrics(slice_data):
    """Calculate metrics for brain region in a slice"""
    # Normalize slice to 0-1 range
    normalized = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-8)
    
    # Apply Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(normalized, sigma=2)
    
    # Use Otsu's method for thresholding
    threshold = filters.threshold_otsu(smoothed)
    binary = smoothed > threshold
    
    # Find connected components
    labels = measure.label(binary)
    regions = measure.regionprops(labels)
    
    # Calculate brain region metrics
    if regions:
        largest_region = max(regions, key=lambda r: r.area)
        brain_area = largest_region.area
    else:
        brain_area = 0
    
    return brain_area

def analyze_brain_regions(img_data):
    """Analyze brain regions across all slices"""
    brain_areas = []
    
    for slice_idx in range(img_data.shape[2]):
        slice_data = img_data[:, :, slice_idx, 0]
        brain_area = calculate_slice_metrics(slice_data)
        brain_areas.append(brain_area)
    
    return brain_areas

def plot_brain_areas(brain_areas, subject_id=None):
    """Plot brain region areas across slices"""
    plt.figure(figsize=(10, 6))
    plt.plot(brain_areas, color='magenta')
    title = 'Brain Region Area'
    if subject_id:
        title += f' - Subject {subject_id}'
    plt.title(title)
    plt.xlabel('Slice Index')
    plt.ylabel('Brain Region Area')
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_pixel_intensities(img_data, slice_idx):
    """Analyze pixel intensity distribution for a slice"""
    slice_data = img_data[:, :, slice_idx, 0]
    pixels = slice_data.flatten()
    
    plt.figure(figsize=(10, 6))
    plt.hist(pixels, bins=50, density=True, alpha=0.7, color='blue')
    plt.title(f'Pixel Intensity Distribution - Slice {slice_idx}')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')
    plt.grid(True, alpha=0.3)
    plt.show()

def load_and_analyze_subject(fs, subject_id):
    """Load and analyze a single subject's data"""
    subject_path = f'oasis-1-dataset-13635/oasis_raw/disc1/{subject_id}/PROCESSED/MPRAGE/T88_111'
    
    try:
        # Get the .img and .hdr files
        img_file = [f for f in fs.ls(subject_path) if f.endswith('.img')][0]
        hdr_file = img_file.replace('.img', '.hdr')
        
        # Download files
        with fs.open(hdr_file, 'rb') as f:
            with open('temp.hdr', 'wb') as out:
                out.write(f.read())
        
        with fs.open(img_file, 'rb') as f:
            with open('temp.img', 'wb') as out:
                out.write(f.read())
        
        # Load the image
        img = nib.load('temp.img')
        img_data = img.get_fdata()
        
        # Analyze brain regions
        brain_areas = analyze_brain_regions(img_data)
        return brain_areas
        
    except Exception as e:
        print(f"Error processing subject {subject_id}: {str(e)}")
        return None

def compare_multiple_subjects(subject_ids):
    """Compare brain region areas across multiple subjects"""
    fs = gcsfs.GCSFileSystem(token='anon')
    all_areas = {}
    
    # Analyze each subject
    for subject_id in subject_ids:
        print(f"Processing {subject_id}...")
        areas = load_and_analyze_subject(fs, subject_id)
        if areas is not None:
            all_areas[subject_id] = areas
    
    # Plot individual subjects
    for subject_id, areas in all_areas.items():
        plot_brain_areas(areas, subject_id)
    
    # Plot all subjects together
    plt.figure(figsize=(12, 8))
    for subject_id, areas in all_areas.items():
        plt.plot(areas, label=subject_id, alpha=0.7)
    plt.title('Brain Region Areas - All Subjects')
    plt.xlabel('Slice Index')
    plt.ylabel('Brain Region Area')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Calculate and plot average brain area
    if all_areas:
        # Convert to numpy arrays for easier calculation
        areas_array = np.array(list(all_areas.values()))
        mean_areas = np.mean(areas_array, axis=0)
        std_areas = np.std(areas_array, axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.plot(mean_areas, 'b-', label='Mean Area')
        plt.fill_between(range(len(mean_areas)), 
                        mean_areas - std_areas, 
                        mean_areas + std_areas, 
                        alpha=0.2, color='blue', 
                        label='±1 Std Dev')
        plt.title('Average Brain Region Area Across Subjects')
        plt.xlabel('Slice Index')
        plt.ylabel('Brain Region Area')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()
        
        # Print slice range recommendations
        peak_mean = np.max(mean_areas)
        threshold = 0.7 * peak_mean  # 70% of peak area
        good_slices = np.where(mean_areas > threshold)[0]
        
        print("\nSlice Range Recommendations:")
        print(f"Peak brain area occurs at slice {np.argmax(mean_areas)}")
        print(f"Recommended slice range (>70% of peak area): {good_slices[0]} to {good_slices[-1]}")

if __name__ == "__main__":
    # Example usage with multiple subjects
    subject_ids = [
        'OAS1_0001_MR1',
        'OAS1_0002_MR1',
        'OAS1_0003_MR1',
        'OAS1_0010_MR1',
        'OAS1_0011_MR1'
    ]
    compare_multiple_subjects(subject_ids)