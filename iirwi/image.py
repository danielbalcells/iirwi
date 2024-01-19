import os
import random
from pathlib import Path

import fastai.vision.all as fv

DEF_PARENT_IMG_PATH = Path('dataset/parents')

def get_crop_coords_from_path(path):
    base_path = os.path.basename(path)
    base_path = os.path.splitext(base_path)[0]
    parts = base_path.split('_')
    coords = parts[-4:]
    coords = [int(coord) for coord in coords]
    return tuple(coords)

def random_crop(input_image, scale=(0.3, 0.4)):
    width, height = input_image.size

    # Calculate random width and height
    new_width = random.randint(int(width * scale[0]), int(width * scale[1]))
    new_height = random.randint(int(height * scale[0]), int(height * scale[1]))

    # Calculate random position for the crop
    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)

    # Perform the crop
    cropped_img = input_image.crop((left, top, left + new_width, top + new_height))

    # Resize the cropped image to 224x224
    resized_img = cropped_img.resize((224, 224))

    # Return the resized image and its coordinates
    return resized_img, (left, top, left + new_width, top + new_height)

class CroppedImage:
    def __init__(self, path, bbox):
        self.path = path
        self.bbox = bbox
    
    @classmethod
    def from_path(cls, path):
        bbox = get_crop_coords_from_path(path)
        return cls(path, bbox)
    
    def get_parent_img_path(self, parent_img_path=DEF_PARENT_IMG_PATH):
        parent_img_name = os.path.basename(self.path).split('_')[0:2]
        parent_img_name = '_'.join(parent_img_name) + '.jpg'
        parent_img_path = os.path.join(parent_img_path, parent_img_name)
        return parent_img_path    

    def load(self):
        return fv.PILImage.create(self.path)

    def load_parent(self):
        return fv.PILImage.create(self.get_parent_img_path())