from typing import List
import torch
from torch import convolution, nn
from torchvision import transforms
from torchvision.models import vgg11_bn, vgg16_bn, VGG11_BN_Weights, VGG16_BN_Weights
from settings.modules.dev import device

def get_vgg11(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT, progress=True)
    else:
        model = vgg11_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    print(model)
    return model.to(device)

def get_vgg16(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT, progress=True)
    else:
        model = vgg16_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model.to(device)

class NoeNet(nn.Module):
    def conv(cls, inc: int, outc: int) -> nn.Conv2d:
        return nn.Conv2d(inc, outc, kernel_size=3, padding=1)

    def mxpool(cls) -> nn.MaxPool2d:
        return nn.MaxPool2d(2, 2, 1)

    def lin(cls, inft: int, outft: int) -> nn.Linear:
        return nn.Linear(inft, outft)

    def drop(cls) -> nn.Dropout:
        return nn.Dropout()

    def __init__(self) -> None:
        super().__init__()
        self.grayscale = transforms.Grayscale()

        self.increase = nn.Sequential(
            self.conv(  1,  32), nn.ReLU(),
            self.conv( 32,  64), nn.ReLU(), self.mxpool(),
            self.conv( 64, 512), nn.ReLU(), self.mxpool()
        )

        self.steady = nn.Sequential(
            self.conv(512, 512), nn.ReLU(), self.mxpool(),
            self.conv(512, 512), nn.ReLU(), self.mxpool()
        )

        self.avgpool = nn.AvgPool2d(2, 2)

        self.classifier = nn.Sequential(
            self.lin(512 * 7 * 7, 1024), nn.ReLU(), self.drop(),
            self.lin(1024, 1024), nn.ReLU(), self.drop(),
            self.lin(1024, 256), nn.ReLU(), self.drop(),
            self.lin(256, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.grayscale(x)
        x = self.increase(x)
        x = self.steady(x)
        x = self.avgpool(x)
        x = x.reshape(-1, 512 * 7 * 7)
        x = self.classifier(x)
        return x

def get_custom() -> nn.Module:
    return NoeNet().to(device)
