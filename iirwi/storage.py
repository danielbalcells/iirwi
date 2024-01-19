import pickle

import torch

from .image import CroppedImage

class ImgFeatureStorage:
    def __init__(self):
        self.features_dict = {}
        self.img_paths = []
        self.images = []

    @classmethod
    def from_features_dict(cls, features_dict):
        obj = cls()
        obj.features_dict = features_dict
        obj.img_paths = list(features_dict.keys())
        obj.images = [CroppedImage.from_path(o) for o in obj.img_paths]
        return obj
    
    def get_features_tensor(self):
        return torch.stack(list(self.features_dict.values()))

    def ix(self, i):
        return self.images[i]

    def export(self, filename):
        pickle.dump(self.features_dict, open(filename, 'wb'))

    @classmethod
    def load(cls, filename):
        features_dict = pickle.load(open(filename, 'rb'))
        return cls.from_features_dict(features_dict)