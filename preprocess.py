import os
from PIL import Image
import random

# Define the source and destination directories
src_dir = Path('dataset4/bboxes')
dst_dir = Path('dataset4/crops3')
dst_dir.mkdir(parents=True, exist_ok=True)

# Define the scale
scale = (0.5, 0.8)

# Loop over all images in the source directory
for img_file in src_dir.glob('*.jpg'):
    # Open the image
    img = Image.open(img_file)
    width, height = img.size

    # Apply the transform 10 times to get 10 random crops
    for i in range(3):
        # Calculate random width and height
        new_width = random.randint(int(width * scale[0]), int(width * scale[1]))
        new_height = random.randint(int(height * scale[0]), int(height * scale[1]))

        # Calculate random position for the crop
        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        # Perform the crop
        cropped_img = img.crop((left, top, left + new_width, top + new_height))

        # Resize the cropped image to 224x224
        resized_img = cropped_img.resize((224, 224))

        # Save the resized image with coordinates in the filename
        resized_img_file = dst_dir / f'{img_file.stem}_crop{i+1}_{left}_{top}_{left+new_width}_{top+new_height}.jpg'
        resized_img.save(resized_img_file)