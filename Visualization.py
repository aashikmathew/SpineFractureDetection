import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_coordinates_from_json(json_path):
    """
    Load coordinates from a JSON file.
    """
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    coordinates = {
        entry['label']: (entry['X'], entry['Y'], entry['Z']) 
        for entry in data if all(key in entry for key in ['X', 'Y', 'Z', 'label'])
    }
    return coordinates

def get_fractured_vertebrae(row, vertebra_mapping):
    """
    Identify fractured vertebrae based on the CSV row and vertebra mapping.
    """
    fractured = []
    for number, label in vertebra_mapping.items():
        fracture_column = f"{label}_fx-g"
        if fracture_column in row.index and row[fracture_column] in [1, 2, 3]:  # Fracture grades 1-3
            fractured.append(number)
    return fractured

def normalize_coordinates(center, dims):
    """
    Normalize coordinates to YOLO format.
    """
    return [(coord / dim) for coord, dim in zip(center, dims)]

def create_bounding_box(center, box_size_mm, voxel_size_mm, dims):
    """
    Create a bounding box of given size around the centroid, converting to voxel coordinates.
    """
    min_coords = [
        int(center[0] - box_size_mm[0] / 2 / voxel_size_mm[0]),
        int(center[1] - box_size_mm[1] / 2 / voxel_size_mm[1]),
        int(center[2] - box_size_mm[2] / 2 / voxel_size_mm[2])
    ]
    max_coords = [
        int(center[0] + box_size_mm[0] / 2 / voxel_size_mm[0]),
        int(center[1] + box_size_mm[1] / 2 / voxel_size_mm[1]),
        int(center[2] + box_size_mm[2] / 2 / voxel_size_mm[2])
    ]
    
    # Ensure bounding box is within image dimensions
    min_coords = [max(0, coord) for coord in min_coords]
    max_coords = [min(dims[i], coord) for i, coord in enumerate(max_coords)]
    
    return min_coords, max_coords

def visualize_fractured_centroids(nii_path, json_path, csv_path, vertebra_mapping, output_path='fractured_vertebrae.png', txt_output_path='fractured_vertebrae.txt', bounding_box_size_mm=[20,20,20]):
    """
    Visualize fractured vertebrae centroids using normalized coordinates and save results to a .txt file.
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load NIfTI image
    nii_img = nib.load(nii_path)
    nii_data = nii_img.get_fdata()
    dims = nii_img.header.get_data_shape()
    voxel_size_mm = nii_img.header.get_zooms()[:3]  # Get voxel size in mm (X, Y, Z)
    
    # Load JSON centroids
    centroids = load_coordinates_from_json(json_path)

    fractured_centroids = set()  # Using set to avoid duplicates
    for _, row in df.iterrows():
        fractured_vertebrae = get_fractured_vertebrae(row, vertebra_mapping)
        for vertebra_number in fractured_vertebrae:
            if vertebra_number in centroids:
                fractured_centroids.add(centroids[vertebra_number])  # Add to set to ensure uniqueness

    # Normalize centroids
    normalized_centroids = [normalize_coordinates(center, dims) for center in fractured_centroids]

    # Save normalized centroids to a .txt file in YOLO format
    with open(txt_output_path, 'w') as file:
        for norm_center in normalized_centroids:
            file.write(f"1 {norm_center[0]:.6f} {norm_center[1]:.6f} {norm_center[2]:.6f}\n")

    # Create bounding boxes for each centroid
    bounding_boxes = []
    for center in fractured_centroids:
        min_coords, max_coords = create_bounding_box(center, bounding_box_size_mm, voxel_size_mm, dims)
        bounding_boxes.append((min_coords, max_coords))
        
        # Visualize the bounding box on the image
        for i in range(3):
            print(f"Bounding Box for Centroid {center}: X:{min_coords[0]} to {max_coords[0]}, Y:{min_coords[1]} to {max_coords[1]}, Z:{min_coords[2]} to {max_coords[2]}")

    # Visualization using normalized coordinates
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Coronal View (x-axis)
    mid_sagittal = dims[0] // 2
    axs[0].imshow(nii_data[mid_sagittal, :, :].T, cmap='gray', origin='lower')
    axs[0].scatter(
        [coord[1] * dims[1] for coord in normalized_centroids],  # Denormalize Y for plotting
        [coord[2] * dims[2] for coord in normalized_centroids],  # Denormalize Z for plotting
        color='red', s=50
    )
    # Draw Bounding Boxes
    for (min_coords, max_coords) in bounding_boxes:
        rect = patches.Rectangle((min_coords[1], min_coords[2]), max_coords[1] - min_coords[1], max_coords[2] - min_coords[2], linewidth=2, edgecolor='blue', facecolor='none')
        axs[0].add_patch(rect)
    axs[0].set_title('Coronal View')

    # Axial View (Y-axis)
    mid_coronal = dims[1] // 2
    axs[1].imshow(nii_data[:, mid_coronal, :].T, cmap='gray', origin='lower')
    axs[1].scatter(
        [coord[0] * dims[0] for coord in normalized_centroids],  # Denormalize X for plotting
        [coord[2] * dims[2] for coord in normalized_centroids],  # Denormalize Z for plotting
        color='red', s=50
    )
    # Draw Bounding Boxes
    for (min_coords, max_coords) in bounding_boxes:
        rect = patches.Rectangle((min_coords[0], min_coords[2]), max_coords[0] - min_coords[0], max_coords[2] - min_coords[2], linewidth=2, edgecolor='blue', facecolor='none')
        axs[1].add_patch(rect)
    axs[1].set_title('Axial View')

    # Sagittal View (Z-axis)
    mid_axial = dims[2] // 2
    axs[2].imshow(np.transpose(nii_data[:, :, mid_axial]), cmap='gray', origin='lower')
    axs[2].scatter(
        [coord[0] * dims[0] for coord in normalized_centroids],  # Denormalize X for plotting
        [coord[1] * dims[1] for coord in normalized_centroids],  # Denormalize Y for plotting
        color='red', s=50
    )
    # Draw Bounding Boxes
    for (min_coords, max_coords) in bounding_boxes:
        rect = patches.Rectangle((min_coords[0], min_coords[1]), max_coords[0] - min_coords[0], max_coords[1] - min_coords[1], linewidth=2, edgecolor='blue', facecolor='none')
        axs[2].add_patch(rect)
    axs[2].set_title('Sagittal View')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    print(f"Visualization saved to {output_path}")
    print(f"Normalized Centroids saved to {txt_output_path}")

# Vertebra mapping for T1-L6
vertebra_mapping = {
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 
    13: 'T6', 14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10', 
    18: 'T11', 19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3', 
    23: 'L4', 24: 'L5', 25: 'L6'
}

# Example usage
visualize_fractured_centroids(
    nii_path='.nii_files/sub-verse004_ct.nii',
    json_path='json_files/sub-verse004_seg-vb_ctd.json',
    csv_path='Annotation_Excel.csv',
    vertebra_mapping=vertebra_mapping,
    output_path='Output_Visualization/fractured_vertebrae_normalized.png',
    txt_output_path='Output_Visualization/fractured_vertebrae_normalized.txt',
    bounding_box_size_mm=[20, 20, 20]
)