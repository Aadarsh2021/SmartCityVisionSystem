import os
import cv2
from realesrgan import RealESRGAN
import torch

# Initialize the RealESRGAN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RealESRGAN(device, scale=4)  # Set the scale factor
model.load_weights('weights/RealESRGAN_x4.pth')  # Replace with the path to your weights file

# Paths for input and output directories
input_dir = 'D:/smart_city_system/data/bdd100k/images/100k/train'  # Update with your dataset path
output_dir = 'D:/smart_city_system/data/bdd100k/images/100k/train_upscaled'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the directory
for root, _, files in os.walk(input_dir):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):  # Process only image files
            input_path = os.path.join(root, file)
            output_path = os.path.join(output_dir, file)
            
            # Load the image
            image = cv2.imread(input_path)
            if image is None:
                print(f"Failed to load image: {input_path}")
                continue

            # Convert image to RGB for RealESRGAN
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Upscale the image
            upscaled_image = model.predict(image)

            # Convert back to BGR for saving
            upscaled_image = cv2.cvtColor(upscaled_image, cv2.COLOR_RGB2BGR)

            # Save the upscaled image
            cv2.imwrite(output_path, upscaled_image)
            print(f"Upscaled and saved: {output_path}")

print("Upscaling complete!")
