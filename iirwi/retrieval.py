import random

import torch

from .feature_extractor import FeatureExtractor
from .storage import ImgFeatureStorage
from .image import random_crop

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
    
    @classmethod
    def from_filenames(cls, extractor_filename, storage_filename, n_crops=DEF_N_CROPS):
        extractor = FeatureExtractor.load(extractor_filename)
        storage = ImgFeatureStorage.load(storage_filename)
        return cls(extractor, storage, n_crops=n_crops)