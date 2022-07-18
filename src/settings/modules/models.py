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
            self.conv( 64, 512), nn.ReLU()
        )

        self.steady = nn.Sequential(
            self.conv(512, 512), nn.ReLU(), self.mxpool(),
            self.conv(512, 512), nn.ReLU(),
            self.conv(512, 512), nn.ReLU(), self.mxpool(),
            self.conv(512, 512), nn.ReLU(),
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
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()

#         self.conv = [
#             nn.Conv2d(3, 64, 3, padding = 1),
#             nn.Conv2d(64, 128, 3, padding = 1),
#             nn.Conv2d(128, 256, 3, padding = 1),
#             nn.Conv2d(256, 512, 3, padding = 1)
#         ]
#         self.conv = nn.ModuleList(self.conv)

#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.3)
#         self.fc = [
#             nn.Linear(7 * 7 * 512, 4096),
#             nn.Linear(4096, 4096),
#             nn.Linear(4096, 1000)
#         ]
#         self.fc = nn.ModuleList(self.fc)
#         self.final = nn.Linear(1000, 2)

#         self.labels_count = 2

#     def forward(self, inputs):
#         x = self.pool(F.relu(self.conv[0](inputs)))

#         for i in range(1, len(self.conv)):
#             x = self.pool(F.relu(self.conv[i](x)))

#         x = self.pool(x)
#         # print(x.shape)

#         x = x.view(-1, 7 * 7 * 512)
#         # print(x.shape)

#         for i in self.fc:
#             x = F.relu(i(x))

#         x = self.dropout(x)
#         x = self.final(x)

#         preds = nn.LogSoftmax(dim = 1)(x)
#         return preds

# def get_model2() -> nn.Module:
#     return Net()