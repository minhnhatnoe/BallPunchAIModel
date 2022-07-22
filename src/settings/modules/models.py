from typing import Tuple
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import vgg11_bn, vgg16_bn, googlenet, alexnet, VGG11_BN_Weights, VGG16_BN_Weights, GoogLeNet_Weights, AlexNet_Weights
from settings.modules.utils import device



def get_googlenet(pretrained: bool) -> Tuple[nn.Module, str]:
    model: nn.Module
    if pretrained:
        name = "google_pr"
        model = googlenet(weights=GoogLeNet_Weights.DEFAULT, progress=True)
    else:
        raise NotImplementedError("No one cares")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return (model.to(device), name)


def get_alexnet(pretrained: bool) -> Tuple[nn.Module, str]:
    model: nn.Module
    if pretrained:
        name = "alexnet_pr"
        model = alexnet(weights=AlexNet_Weights.DEFAULT, progress=True)
    else:
        raise NotImplementedError("No one cares")
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return (model.to(device), name)



def get_vgg11(pretrained: bool) -> Tuple[nn.Module, str]:
    model: nn.Module
    name: str
    if pretrained:
        name = "vgg11_bn_pr"
        model = vgg11_bn(weights=VGG11_BN_Weights.DEFAULT, progress=True)
    else:
        name = "vgg11_bn_sc"
        model = vgg11_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return (model.to(device), name)

def get_vgg16(pretrained: bool) -> Tuple[nn.Module, str]:
    model: nn.Module
    if pretrained:
        name = "vgg16_bn_pr"
        model = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT, progress=True)
    else:
        name = "vgg16_bn_sc"
        model = vgg16_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return (model.to(device), name)



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
            self.conv(  1,  32), nn.ReLU(), self.mxpool(),
            self.conv( 32,  64), nn.ReLU(),
            self.conv( 64, 128), nn.ReLU(), self.mxpool(),
            self.conv(128, 256), nn.ReLU(),
            self.conv(256, 512), nn.ReLU(), self.mxpool()
        )

        self.steady = nn.Sequential(
            self.conv(512, 512), nn.ReLU(),
            # self.conv(512, 512), nn.ReLU(),
            self.conv(512, 512), nn.ReLU(), self.mxpool()
        )

        self.avgpool = nn.AvgPool2d(2, 2)

        self.classifier = nn.Sequential(
            self.lin(512 * 7 * 7, 256), nn.ReLU(), self.drop(),
            self.lin(256, 256), nn.ReLU(), self.drop(),
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

def get_custom() -> Tuple[nn.Module, str]:
    return (NoeNet().to(device), "custom")
