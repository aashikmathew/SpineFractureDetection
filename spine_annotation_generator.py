import os
import json
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class SpineAnnotationGenerator:
    def __init__(self, nii_folder, json_folder, csv_path, output_folder):
        """
        Initialize the annotation generator
        """
        self.nii_folder = Path(nii_folder)
        self.json_folder = Path(json_folder)
        self.csv_path = csv_path
        self.output_folder = Path(output_folder)

        # Create output folder if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)

        # Read CSV file
        self.df = pd.read_csv(csv_path)

        # Vertebra mapping
        self.vertebra_mapping = {
            8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5',
            13: 'T6', 14: 'T7', 15: 'T8', 16: 'T9', 17: 'T10',
            18: 'T11', 19: 'T12', 20: 'L1', 21: 'L2', 22: 'L3',
            23: 'L4', 24: 'L5', 25: 'L6'
        }

        print("All columns in CSV:", list(self.df.columns))

    def load_json_centroids(self, json_path):
        """
        Load centroid coordinates from JSON file
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        centroids = {entry['label']: (entry['X'], entry['Y'], entry['Z'])
                     for entry in data if 'label' in entry}
        return centroids

    def normalize_coordinates(self, nii_file, raw_coords, raw_size):
        """
        Normalize coordinates and dimensions to YOLO format (0 to 1)
        """
        nii_img = nib.load(nii_file)
        dims = nii_img.header.get_data_shape()

        # Normalize each dimension
        normalized_coords = [raw_coords[i] / dims[i] for i in range(3)]
        normalized_size = [raw_size[i] / dims[i] for i in range(3)]

        return normalized_coords, normalized_size
    
    def calculate_bounding_box_size(centroids, vertebra_number):
        neighbors = [vertebra_number - 1, vertebra_number + 1]
        adjacent = [centroids[label] for label in neighbors if label in centroids]
        if len(adjacent) < 2:
            return (30, 30, 30)  # Default size
        dx = max(abs(centroids[vertebra_number][0] - adj[0]) for adj in adjacent)
        dy = max(abs(centroids[vertebra_number][1] - adj[1]) for adj in adjacent)
        dz = max(abs(centroids[vertebra_number][2] - adj[2]) for adj in adjacent)
        return (dx * 2, dy * 2, dz * 2)

    def visualize_sample(self, nii_file, annotations):
        """
        Visualize a sample NIfTI scan with YOLO bounding boxes overlaid
        """
        nii_img = nib.load(nii_file)
        img_data = nii_img.get_fdata()
        depths, heights, widths = img_data.shape

        # Default slice indices
        slice_indices = [depths // 2, heights // 2, widths // 2]

        fig = plt.figure(figsize=(15, 5))
        titles = ['Axial Slice', 'Coronal Slice', 'Sagittal Slice']
        planes = [
            img_data[slice_indices[0], :, :],  # Axial (Z-slice)
            img_data[:, slice_indices[1], :],  # Coronal (Y-slice)
            np.transpose(img_data[:, :, slice_indices[2]])  # Sagittal (X-slice, rotated vertically)
        ]

        for i, (plane, title) in enumerate(zip(planes, titles)):
            ax = fig.add_subplot(1, 3, i + 1)
            ax.imshow(plane, cmap='gray')
            ax.set_title(title)

            # Visualize bounding boxes
            for ann in annotations:
                cls, cx, cy, cz, w, h, d = map(float, ann.split())
                if i == 0 and abs(cz - slice_indices[0] / depths) <= d / 2:
                    rect = patches.Rectangle(
                        (cx * widths - w * widths / 2, cy * heights - h * heights / 2),
                        w * widths, h * heights,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)

                elif i == 1 and abs(cy - slice_indices[1] / heights) <= h / 2:
                    rect = patches.Rectangle(
                        (cx * widths - w * widths / 2, cz * depths - d * depths / 2),
                        w * widths, d * depths,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)

                elif i == 2 and abs(cx - slice_indices[2] / widths) <= w / 2:
                    rect = patches.Rectangle(
                        (cz * depths - d * depths / 2, cy * heights - h * heights / 2),
                        d * depths, h * heights,
                        linewidth=2, edgecolor='lime', facecolor='none'
                    )
                    ax.add_patch(rect)

        plt.tight_layout()
        plt.show()

    def generate_annotations(self):
        """
        Generate YOLO annotations for vertebrae in the dataset
        """
        for _, row in self.df.iterrows():
            verse_id = row['verse_ID']
            nii_file_name = f"sub-verse{verse_id:03d}_ct.nii"
            nii_file = list(self.nii_folder.glob(nii_file_name))

            json_file_name = nii_file_name.replace('_ct.nii', '_seg-vb_ctd.json')
            json_file = list(self.json_folder.glob(json_file_name))

            if not nii_file or not json_file:
                print(f"Skipping {nii_file_name}: Matching NIfTI or JSON file not found")
                continue

            nii_file = nii_file[0]
            json_file = json_file[0]

            print(f"Processing NIfTI: {nii_file}")
            print(f"Processing JSON: {json_file}")

            centroids = self.load_json_centroids(json_file)
            annotation_content = []

            for vertebra_number, vertebra_label in self.vertebra_mapping.items():
                fracture_column = f"{vertebra_label}_fx-g"

                if vertebra_number in centroids:
                    class_number = 0
                    if fracture_column in row.index and row[fracture_column] in [1, 2, 3]:
                        class_number = 1

                    center = centroids[vertebra_number]
                    box_size = (30, 30, 30)  # Default bounding box size for simplicity

                    normalized_center, normalized_box_size = self.normalize_coordinates(
                        nii_file, center, box_size
                    )

                    annotation_line = (
                        f"{class_number} {normalized_center[2]:.6f} {normalized_center[0]:.6f} "
                        f"{normalized_center[1]:.6f} {normalized_box_size[2]:.6f} "
                        f"{normalized_box_size[0]:.6f} {normalized_box_size[1]:.6f}"
                    )

                    if class_number == 1:
                        annotation_content.append(annotation_line)

            if annotation_content:
                output_file = self.output_folder / f"sub-verse{verse_id:03d}_ct.nii_annotations.txt"
                with open(output_file, 'w') as f:
                    f.write('\n'.join(annotation_content))
                print(f"Generated annotations for verse_ID {verse_id}")


def main():
    generator = SpineAnnotationGenerator(
        nii_folder='.nii_files',
        json_folder='json_files',
        csv_path='Annotation_Excel.csv',
        output_folder='output_annotation'
    )
    generator.generate_annotations()


if __name__ == "__main__":
    main()
