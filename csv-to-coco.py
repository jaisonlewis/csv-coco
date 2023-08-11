#this is made specifically for the data I have it is not a generic CSV to COCO convertor

import json
import os
from skimage import io
import csv

# Dataset directories
images_dir = 'kidney_train'
csv_file = 'annotations-new.csv'

coco_format = {
    'images': [],
    'categories': [
        {'name': 'glomerulus', 'id': 0},
        {'name': 'blood_vessel', 'id': 1},
        {'name': 'unsure', 'id': 2},
    ],
    'annotations': []
}

# To ensure uniqueness of image entries and map filenames to new integer IDs
image_filename_to_id = {}
current_image_id = 1
current_annotation_id = 1

# Parse CSV file using csv.reader to handle embedded newlines and other quirks
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip the header line
    for fields in reader:
        # Filename
        filename = fields[0]

        # Determine the category_id based on the category_name
        category_name = fields[-1].split(':')[-1].strip().replace('"', '').replace('}', '')
        if category_name == 'glomerulus':
            category_id = 0
        elif category_name == 'blood_vessel':
            category_id = 1
        else:
            category_id = 2

        # Get image ID, assign a new one if this image hasn't been processed yet
        if filename not in image_filename_to_id:
            image_id = current_image_id
            image_filename_to_id[filename] = image_id
            current_image_id += 1
        else:
            image_id = image_filename_to_id[filename]

        # Image path
        image_path = os.path.join(images_dir, filename)

        # Read image to get its dimensions
        image = io.imread(image_path)
        height, width = image.shape[:2]

        # Add image to the 'images' list only if it hasn't been added already
        if image_id == len(coco_format['images']) + 1:
            coco_format['images'].append({
                'file_name': filename,
                'height': height,
                'width': width,
                'id': image_id,
            })

        # Process annotations
        json_string = fields[5].strip()

        # Remove leading and trailing quotes if they exist
        if json_string.startswith('"') and json_string.endswith('"'):
            json_string = json_string[1:-1]

        # Replace consecutive double quotes with a single double quote
        json_string = json_string.replace('""', '"')

        try:
            polygon_data = json.loads(json_string)
            all_points_x = polygon_data["all_points_x"]
            all_points_y = polygon_data["all_points_y"]

            # Compute bounding box
            x_min, x_max = min(all_points_x), max(all_points_x)
            y_min, y_max = min(all_points_y), max(all_points_y)
            bbox_width, bbox_height = x_max - x_min, y_max - y_min
            bbox = [x_min, y_min, bbox_width, bbox_height]

            # Compute area (using bounding box as an approximation)
            area = bbox_width * bbox_height

            # Create segmentation data
            segmentation = [list(zip(all_points_x, all_points_y))]
            segmentation = [coord for sublist in segmentation for point in sublist for coord in
                            point]  # Flatten the list

            annotation = {
                'iscrowd': 0,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': [segmentation],  # COCO expects a list of lists
                'bbox': bbox,
                'area': area,
                'id': current_annotation_id  # Assign unique ID to each annotation
            }

            coco_format['annotations'].append(annotation)

            # Increment the annotation ID for the next one
            current_annotation_id += 1
        except json.JSONDecodeError:
            print(f"Error decoding JSON for line: {line}")
            print(f"Problematic JSON string: {json_string}")
            continue

with open('coco_annotations.json', 'w') as f:
    json.dump(coco_format, f)
