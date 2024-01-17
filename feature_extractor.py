import os
import pickle
import torch
import matplotlib.pyplot as plt
from fastai.vision.all import *


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

def train_model(dls, n_epochs=5):
    learn = vision_learner(dls, resnet18, metrics=error_rate)
    learn.fine_tune(n_epochs)
    return learn

def get_feature_extractor(learn, dls, loss_func=dummy_loss_func):
    model = FeatureExtractor(learn.model)
    feature_extractor = Learner(dls, model, loss_func=loss_func)
    return feature_extractor

def train_feature_extractor(dataset_path, item_tfms=[Resize(224)], label_func=get_label, n_epochs=5):
    path = Path(dataset_path)
    dls = ImageDataLoaders.from_name_func(
        path, get_image_files(path), valid_pct=0.2, seed=42,
        label_func=get_label, item_tfms=item_tfms)   
    learn = train_model(dls, n_epochs=n_epochs)
    feature_extractor = get_feature_extractor(learn, dls)
    return feature_extractor

def export_feature_extractor(feature_extractor, model_name):
    feature_extractor.path = Path('.')
    feature_extractor.export(model_name)

def load_feature_extractor(model_name):
    feature_extractor = load_learner(model_name)
    return feature_extractor
def get_dataset_features(feature_extractor):
    dls = feature_extractor.dls
    # Get activations for training data
    train_activations, _ = feature_extractor.get_preds(dl=dls.train)
    # Get activations for validation data
    valid_activations, _ = feature_extractor.get_preds(dl=dls.valid)
    # Concatenate the activations
    all_activations = torch.cat([train_activations, valid_activations])
    # Concatenate the image paths
    all_items = dls.train.items + dls.valid.items
    # Create a dictionary mapping image paths to features
    features = {image: activation.clone() for image, activation in zip(all_items, all_activations)}
    return features

def write_features(features, filename):
    with open(filename, 'wb') as f:
        pickle.dump(features, f)

def load_features(filename):
    with open(filename, 'rb') as f:
        features = pickle.load(f)
    return features

def get_features_tensor_from_dict(features):
    # Convert the features dictionary to a list of tuples
    features_list = list(features.items())
    # Extract the image paths and features
    image_paths, feature_tensors = zip(*features_list)
    # Convert the features to a PyTorch tensor
    features_tensor = torch.stack(feature_tensors)
    return features_tensor, image_paths