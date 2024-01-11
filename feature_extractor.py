from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        return x.view(x.size(0), -1)