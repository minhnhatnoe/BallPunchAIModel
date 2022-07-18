from torch import nn
from torchvision.models import vgg11_bn, vgg16_bn, VGG11_BN_Weights, VGG16_BN_Weights

def get_vgg11(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg11_bn(weights=VGG11_BN_Weights, progress=True)
    else:
        model = vgg11_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model

def get_vgg16(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg16_bn(weights=VGG16_BN_Weights, progress=True)
    else:
        model = vgg16_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model
    
def get_custom() -> nn.Module:
    pass
