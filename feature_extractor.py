import os
from pathlib import Path
from fastai.vision.all import *
from torch import nn

# Define a new model that includes a global max pooling layer
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return x.view(x.size(0), -1)
    
def dummy_loss_func(x, y):
    return torch.tensor(0.)

def get_label(file_path):
        return os.path.basename(file_path).split('_')[0]

def train_model(dataset_path, item_tfms=None, label_func=get_label):
    path = Path(dataset_path)
    
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=get_label, item_tfms=item_tfms)   

    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(5)
    return learn

def get_feature_extractor(learn, loss_func=dummy_loss_func):
    model = FeatureExtractor(learn.model)
    feature_extractor = Learner(learn.dls, model, loss_func=loss_func)
    return feature_extractor

def train_feature_extractor(dataset_path, item_tfms=None, label_func=get_label):
    learn = train_model(dataset_path, item_tfms=item_tfms, label_func=label_func)
    feature_extractor = get_feature_extractor(learn)
    return feature_extractor