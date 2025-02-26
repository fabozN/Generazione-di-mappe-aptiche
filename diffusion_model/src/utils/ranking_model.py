import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, resnet50
import numpy as np


class RankingEncoder(nn.Module):
    def __init__(self, feature_dim=32,
                 model_type='resnet18',
                 **kwargs):
        super().__init__()
        if model_type == 'resnet18':
            self.resnet = resnet18(pretrained=True)
            self.resnet.fc = nn.Linear(512, feature_dim)
        elif model_type == 'resnet50':
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Linear(2048, feature_dim)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        feat_norm = torch.norm(features, dim=1)
        features /= feat_norm.view(features.shape[0], 1)
        return features

    def encode(self, im: np.ndarray):
        '''
        takes in an image and returns the resnet18 features
        '''
        im = im.unsqueeze(0)
        with torch.no_grad():
            features = self(im)
        return features

    def save(self, save_name):
        torch.save(self.state_dict(), save_name)

    def load(self, save_name):
        self.load_state_dict(torch.load(
            save_name, map_location=torch.device('cpu')))