import torch
import torch.nn as nn
import torchvision.models as models

class MultitaskFaceModel(nn.Module):
    def __init__(self):
        super(MultitaskFaceModel, self).__init__()
        
        # We use MobileNetV2 as our pre-trained backbone (updated to modern weights syntax)
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Gender Head (Binary classification)
        self.gender_head = nn.Linear(1280, 1)
        
        # Hair Head (4-class classification)
        self.hair_head = nn.Linear(1280, 4)
        
        # Age Head (Binary classification)
        self.age_head = nn.Linear(1280, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        gender = torch.sigmoid(self.gender_head(x))
        hair = torch.softmax(self.hair_head(x), dim=1)
        age = torch.sigmoid(self.age_head(x))
        
        return gender, hair, age

def get_device():
    # Automatically routes tensor operations to your RTX card
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')