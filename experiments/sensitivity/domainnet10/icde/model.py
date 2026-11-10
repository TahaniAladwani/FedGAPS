import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class Network(nn.Module):
    def __init__(self, classes, backbone='resnet18', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            self.encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.feature_dim = self.encoder.fc.in_features
            self.encoder.fc = nn.Identity()
        else:
            self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(), nn.AvgPool2d(2), nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.feature_dim = 32
        self.classifier = nn.Linear(self.feature_dim, classes)

    def forward(self, images):
        features = self.encoder(images)
        return features, self.classifier(features)

    def bn_keys(self):
        keys = set()
        for name, module in self.named_modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                keys.update(f'{name}.{key}' for key in module.state_dict())
        return keys
