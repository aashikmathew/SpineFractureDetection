import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_yolo_annotation(annotation_line):
    """
    Parse YOLO format annotation line
    
    Format: class x_center y_center z_center x_length y_length z_length
    All coordinates are normalized (0-1 or -0.5 to 1.5)
    """
    parts = list(map(float, annotation_line.split()))
    return {
        'class': parts[0],
        'center_x': parts[1],
        'center_y': parts[2],
        'center_z': parts[3],
        'width_x': parts[4],
        'width_y': parts[5],
        'width_z': parts[6]
    }

def visualize_3d_annotations(nii_path, annotation_path):
    """
    Visualize 3D image with dot annotations
    
    :param nii_path: Path to NIfTI image file
    :param annotation_path: Path to annotation file
    """
    # Load NIfTI image
    nii_img = nib.load(nii_path)
    img_data = nii_img.get_fdata()
    
    # Get image dimensions
    depths, heights, widths = img_data.shape
    
    # Read annotations
    with open(annotation_path, 'r') as f:
        annotations = [parse_yolo_annotation(line.strip()) for line in f]
    
    # Visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Original image slices
    slice_indices = [
        img_data.shape[0] // 2,  # Axial slice
        img_data.shape[1] // 2,  # Coronal slice
        img_data.shape[2] // 2   # Sagittal slice
    ]
    
    slice_planes = [
        img_data[slice_indices[0], :, :],  # Axial
        img_data[:, slice_indices[1], :],  # Coronal
        np.transpose(img_data[:, :, slice_indices[2]])   # Transpose Sagittal to make it vertical
    ]
    
    titles = ['Coronal', 'Axial', 'Sagittal']
    
    for i, (slice_data, title) in enumerate(zip(slice_planes, titles)):
        ax = fig.add_subplot(1, 3, i+1)
        ax.imshow(slice_data, cmap='gray')
        ax.set_title(f'{title} Slice')
        
        # Add dot annotations for this slice
        for ann in annotations:
            # Convert normalized coordinates back to pixel coordinates
            center_x = ann['center_x'] * widths
            center_y = ann['center_y'] * heights
            center_z = ann['center_z'] * depths
            
            # Color-code based on class (0: normal, 1: fracture)
            color = 'lime' if ann['class'] == 1 else 'red'
            
            # Determine which slice to draw based on current plane
            if title == 'Axial':
                # Draw using Coronal slice points (center_x, center_z)
                ax.scatter(center_x, center_y, color=color, marker='x', s=100, linewidth=2)
            elif title == 'Coronal':
                # Draw using Axial slice points (center_x, center_y)
                ax.scatter(center_x, center_y, color=color, marker='x', s=100, linewidth=2)
            else:  # Sagittal (now vertical)
                # Draw on X-slice (side view, now rotated)
                ax.scatter(center_z, center_y, color=color, marker='x', s=100, linewidth=2)
    
    plt.tight_layout()
    plt.show()

def main():
    nii_path = '.nii_files/sub-verse004_ct.nii'
    annotation_path = 'output_annotation/sub-verse004_ct.nii_annotations.txt'
    visualize_3d_annotations(nii_path, annotation_path)

if __name__ == "__main__":
    main()