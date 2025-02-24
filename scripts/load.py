import os
import json
from PIL import Image

# Updated class map for the specified categories
class_map = {
    'Car': 0,
    'Bus': 1,
    'Truck': 2,
    'Motorcycle': 3,
    'Bicycle': 4,
    'Pedestrian': 5,  # This corresponds to 'other person'
    'Other person': 6,  # Added mapping for 'Other person'
    'Rider': 7,
    'Traffic light': 8,
    'Traffic sign': 9,
    'Trailer': 10,
    'Train': 11,
    'Other vehicle': 12,  # Optional mapping for 'other vehicle'
}

def normalize_category_name(category):
    """Normalize category names for consistent mapping."""
    return category.strip().capitalize()

def convert_bdd100k_to_yolo(json_path, images_dir, output_dir):
    with open(json_path) as f:
        data = json.load(f)

    for item in data:
        image_name = item['name']
        image_path = os.path.join(images_dir, image_name)

        # Ensure the image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} does not exist. Skipping...")
            continue
        
        # Retrieve image dimensions using PIL
        try:
            with Image.open(image_path) as img:
                image_width, image_height = img.size
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            continue

        labels = item.get('labels', [])
        
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Prepare output file for YOLO format
        label_file = os.path.join(output_dir, image_name.replace('.jpg', '.txt'))
        with open(label_file, 'w') as f:
            for label in labels:
                category = normalize_category_name(label['category'])  # Normalize category
                if category in class_map:
                    class_id = class_map[category]
                    bbox = label.get('box2d')
                    if bbox:
                        x_center = (bbox['x1'] + bbox['x2']) / 2.0 / image_width
                        y_center = (bbox['y1'] + bbox['y2']) / 2.0 / image_height
                        width = (bbox['x2'] - bbox['x1']) / image_width
                        height = (bbox['y2'] - bbox['y1']) / image_height
                        f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
                    else:
                        print(f"Warning: No bounding box found for label {label['id']} in {image_name}.")
                else:
                    print(f"Warning: Category '{category}' not found in class map for image {image_name}.")

def convert_datasets(train_json_path, train_images_dir, train_output_dir, val_json_path, val_images_dir, val_output_dir):
    # Convert the training dataset
    print("Converting training set...")
    convert_bdd100k_to_yolo(train_json_path, train_images_dir, train_output_dir)
    
    # Convert the validation dataset
    print("Converting validation set...")
    convert_bdd100k_to_yolo(val_json_path, val_images_dir, val_output_dir)

# Paths for the train and validation sets
train_json_path = 'D:/smart_city_system/data/bdd100k/labels/det_20/det_train.json'
train_images_dir = 'D:/smart_city_system/data/bdd100k/images/100k/train'
train_output_dir = 'D:/smart_city_system/yolo_labels/train'

val_json_path = 'D:/smart_city_system/data/bdd100k/labels/det_20/det_val.json'
val_images_dir = 'D:/smart_city_system/data/bdd100k/images/100k/val'
val_output_dir = 'D:/smart_city_system/yolo_labels/val'

# Convert both train and val datasets to YOLO format
convert_datasets(train_json_path, train_images_dir, train_output_dir, val_json_path, val_images_dir, val_output_dir)
