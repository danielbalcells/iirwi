import random

import torch

from feature_extractor import FeatureExtractor
from storage import ImgFeatureStorage
from image import random_crop

DEF_N_CROPS = 10

class CroppingImgRetriever:
    def __init__(self, extractor, storage, n_crops=DEF_N_CROPS):
        self.extractor = extractor
        self.storage = storage
        self.n_crops = n_crops
    
    def process(self, input_img):
        max_similarity = -1
        most_similar_img = None
        input_img_crop_coords = None
        for i in range(self.n_crops):
            cropped_img, crop_coords = random_crop(input_img)
            similar_img, similarity = self.get_similar_img(cropped_img)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_img = similar_img
                input_img_crop_coords = crop_coords
        return most_similar_img, input_img_crop_coords

    def get_similar_img(self, input_img):
        input_features = self.extractor.predict(input_img)
        storage_features = self.storage.get_features_tensor()
        similarities = torch.nn.functional.cosine_similarity(input_features, storage_features)
        most_similar_index = torch.argmax(similarities)
        similar_img = self.storage.ix(most_similar_index)
        return similar_img, similarities[most_similar_index]

        

# def get_image_features(input_image, feature_extractor):
#     with feature_extractor.no_bar(), feature_extractor.no_logging():
#         test_dl = feature_extractor.dls.test_dl([input_image])
#         inp, features, _, dec = feature_extractor.get_preds(dl=test_dl, with_input=True, with_decoded=True)
#     return features

# def get_similar_image(input_image, feature_extractor, features_tensor, image_paths):
#     # Get the features of the input image
#     user_features = get_image_features(input_image, feature_extractor)
#     user_features = user_features.view(1, -1)  # Reshape to 2D tensor
#     # Compute cosine similarity
#     similarity_scores = torch.nn.functional.cosine_similarity(user_features, features_tensor)
#     # Get the index of the most similar image
#     most_similar_index = torch.argmax(similarity_scores)
#     # Get the path of the most similar image
#     most_similar_image_path = image_paths[most_similar_index]
#     # Get the maximum similarity score
#     max_similarity = torch.max(similarity_scores)
#     return most_similar_image_path, max_similarity

# def plot_side_by_side(input_image, similar_image, show=True, save_path=None):
#     similar_image_thumb = similar_image.to_thumb(224)
#     user_image_thumb = input_image.to_thumb(224)
#     # Create a figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2)

#     # Display the images
#     ax1.imshow(similar_image_thumb)
#     ax2.imshow(user_image_thumb)

#     # Optionally, remove the axes for a cleaner look
#     ax1.axis('off')
#     ax2.axis('off')

#     fig.suptitle('Is It Really Worth It?', fontsize=20, weight='bold')
#     if save_path:
#          plt.savefig(save_path)
#          plt.close()
#     if show:
#         plt.show()

# def test_model(feature_extractor, features_tensor, model_name, image_paths, input_dir=Path('input'), output_dir=Path('output'), show=False):
#     save_dir = output_dir / model_name
#     save_dir.mkdir(parents=True, exist_ok=True)
#     for input_path in input_dir.iterdir():
#         save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(input_path))[0] + '.jpg')
#         input_image = PILImage.create(input_path)
#         process_image(input_image, feature_extractor, features_tensor, image_paths, save_path=save_path, show=show)

# def random_crop(input_image, scale=(0.3, 0.4)):
#     width, height = input_image.size

#     # Calculate random width and height
#     new_width = random.randint(int(width * scale[0]), int(width * scale[1]))
#     new_height = random.randint(int(height * scale[0]), int(height * scale[1]))

#     # Calculate random position for the crop
#     left = random.randint(0, width - new_width)
#     top = random.randint(0, height - new_height)

#     # Perform the crop
#     cropped_img = input_image.crop((left, top, left + new_width, top + new_height))

#     # Resize the cropped image to 224x224
#     resized_img = cropped_img.resize((224, 224))

#     # Return the resized image and its coordinates
#     return resized_img, (left, top, left + new_width, top + new_height)

# def process_image(input_image, feature_extractor, features_tensor, image_paths, show=True, save_path=None):
#     max_similarity = -1
#     most_similar_image_path = None
#     input_image_crop_coords = None
#     reference_image_crop_coords = None

#     # Apply the transform 10 times to get 10 random crops
#     for i in range(10):
#         # Perform a random crop
#         cropped_img, crop_coords = random_crop(input_image)

#         # Get the most similar image for the cropped image and its similarity score
#         similar_image_path, similarity = get_similar_image(cropped_img, feature_extractor, features_tensor, image_paths)

#         # If this image is more similar than the previous ones, keep it
#         if similarity > max_similarity:
#             max_similarity = similarity
#             most_similar_image_path = similar_image_path
#             input_image_crop_coords = crop_coords
#             reference_image_crop_coords = get_crop_coords_from_filename(similar_image_path)

#     # Get the parent and crop coordinates from the filename
#     parent, filename = os.path.split(most_similar_image_path)

#     # Plot the input image and the most similar image side by side
#     plot_side_by_side(input_image, PILImage.create(most_similar_image_path), input_image_crop_coords, reference_image_crop_coords, show=show, save_path=save_path)

#     return parent, filename, input_image_crop_coords, reference_image_crop_coords