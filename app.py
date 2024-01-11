import pickle
from functools import partial
from pathlib import Path
from io import BytesIO


import matplotlib.pyplot as plt
import torch
from fastai.vision.all import *
import gradio as gr


from feature_extractor import FeatureExtractor


MODEL_NAME = Path('model_dataset2-3_smallcrop_tinyresize.pkl')
FEATURES_NAME = Path('features_dataset2-3_smallcrop_tinyresize.pkl')

def get_label(file_path):
    return os.path.basename(file_path).split('_')[0]

def loss_func(x, y):
    return torch.tensor(0.)

def get_image_features(input_image, feature_extractor):
        with feature_extractor.no_bar(), feature_extractor.no_logging():
            _, features, _ = feature_extractor.predict(input_image)
        return features

def get_similar_image(input_image, feature_extractor, features_dict):
    # Convert the features dictionary to a list of tuples
    features_list = list(features_dict.items())

    # Extract the image paths and features
    image_paths, feature_tensors = zip(*features_list)

    # Convert the features to a PyTorch tensor
    features_tensor = torch.stack(feature_tensors)

    # Now, to compute the cosine similarity between the user's input image and all other images:
    user_features = get_image_features(input_image, feature_extractor)
    user_features = user_features.view(1, -1)  # Reshape to 2D tensor

    # Compute cosine similarity
    similarity_scores = torch.nn.functional.cosine_similarity(user_features, features_tensor)

    # Get the index of the most similar image
    most_similar_index = torch.argmax(similarity_scores)

    # Get the path of the most similar image
    most_similar_image_path = image_paths[most_similar_index]
    # Display the most similar image and the input image side by side
    most_similar_image = PILImage.create(most_similar_image_path)
    return most_similar_image

def plot_side_by_side(input_image, similar_image, show=True, save_path=None):
    similar_image_thumb = similar_image.to_thumb(224)
    user_image_thumb = input_image.to_thumb(224)
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # Display the images
    ax1.imshow(similar_image_thumb)
    ax2.imshow(user_image_thumb)

    # Optionally, remove the axes for a cleaner look
    ax1.axis('off')
    ax2.axis('off')

    fig.suptitle('Is It Really Worth It?', fontsize=20, weight='bold')
    if save_path:
         plt.savefig(save_path)
         plt.close()
    if show:
        plt.show()
    # Convert the plot to a PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    result_image = Image.open(buf)

    return result_image

def process_image(input_image, feature_extractor, features_dict, show=True, save_path=None):
    similar_image = get_similar_image(input_image, feature_extractor, features_dict)
    meme = plot_side_by_side(input_image, similar_image, show=show, save_path=save_path)
    return meme

def load_model_and_features():
    # Load the model
    feature_extractor = load_learner(MODEL_NAME)

    with open(FEATURES_NAME, 'rb') as f:
        features_dict = pickle.load(f)

    return feature_extractor, features_dict

def predict(input_image):
     img = PILImage.create(input_image)
     feature_extractor, features_dict = load_model_and_features()
     return process_image(img, feature_extractor, features_dict, show=False)
    
iface = gr.Interface(
    fn=predict,
    inputs='image',
    outputs='image',
)

iface.launch(debug=True)