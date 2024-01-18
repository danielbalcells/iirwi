import os
import pickle
from pathlib import Path
from collections.abc import Iterable

import torch
import matplotlib.pyplot as plt
import fastai.vision.all as fv
import torch.nn as nn

from storage import ImgFeatureStorage

def dummy_loss_func(x, y):
    return torch.tensor(0.)

def get_label(file_path):
        return os.path.basename(file_path).split('_')[0]

class FeatureExtractorModel(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return x.view(x.size(0), -1)
    
class FeatureExtractor:
    def __init__(self, dataset_path=None, dls=None, item_tfms=None, label_func=get_label, n_epochs=5):
        item_tfms = item_tfms or [fv.Resize(224)]
        self.dataset_path = dataset_path
        self.dls = dls
        self.item_tfms = item_tfms
        self.label_func = label_func
        self.n_epochs = n_epochs
        if self.dataset_path and not self.dls:
            self.dls = fv.ImageDataLoaders.from_name_func(
                self.dataset_path, fv.get_image_files(self.dataset_path), valid_pct=0.2, seed=42,
                label_func=self.label_func, item_tfms=self.item_tfms)
    
    @classmethod
    def from_dataset(cls, dataset_path, item_tfms=[fv.Resize(224)], label_func=get_label, n_epochs=5):
        return cls(dataset_path=dataset_path, item_tfms=item_tfms, label_func=label_func, n_epochs=n_epochs)

    @classmethod
    def from_learner(cls, extractor):
        obj = cls(dls=extractor.dls)
        obj.extractor = extractor
        return obj

    @classmethod
    def load(cls, filename, label_func=get_label, item_tfms=[fv.Resize(224)]):
        extractor = fv.load_learner(filename, cpu=False)
        dls = fv.ImageDataLoaders.from_name_func(
            extractor.dls.path, fv.get_image_files(extractor.dls.path), valid_pct=0.2, seed=42,
            label_func=label_func, item_tfms=item_tfms)
        extractor.dls = dls
        return cls.from_learner(extractor)

    def export(self, model_name, path=Path('.')):
        self.extractor.path = path
        self.extractor.export(model_name)
    
    def train(self, n_epochs=None):
        n_epochs = n_epochs or self.n_epochs
        self.classifier = self.train_classifier(n_epochs)
        self.extractor = self.get_extractor()

    def train_classifier(self, n_epochs=None):
        n_epochs = n_epochs or self.n_epochs
        classifier = fv.vision_learner(self.dls, fv.resnet18, metrics=fv.error_rate)
        classifier.fine_tune(n_epochs)
        return classifier
    
    def get_extractor(self):
        model = FeatureExtractorModel(self.classifier.model)
        extractor = fv.Learner(self.dls, model, loss_func=dummy_loss_func)
        return extractor

    def predict(self, input_images):
        if not isinstance(input_images, Iterable) or isinstance(input_images, str):
            input_images = [input_images]
        with self.extractor.no_bar(), self.extractor.no_logging():
            dl = self.extractor.dls.test_dl(input_images)
            inp, features, _, dec = self.extractor.get_preds(dl=dl, with_input=True, with_decoded=True)
        return features
    
    def predict_for_dataset(self, dls=None):
        dls = dls or self.dls
        train_features, _ = self.extractor.get_preds(dl=dls.train)
        valid_features, _ = self.extractor.get_preds(dl=dls.valid)
        all_features = torch.cat([train_features, valid_features])
        all_items = dls.train.items + dls.valid.items
        # Create a dictionary mapping image paths to features
        features = {image: activation.clone() for image, activation in zip(all_items, all_features)}
        return ImgFeatureStorage.from_features_dict(features)