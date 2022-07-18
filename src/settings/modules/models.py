from torch import nn
from torchvision.models import vgg11_bn, vgg16_bn, VGG11_BN_Weights, VGG16_BN_Weights
from dev import device

def get_vgg11(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg11_bn(weights=VGG11_BN_Weights, progress=True)
    else:
        model = vgg11_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model.to(device)

def get_vgg16(pretrained: bool) -> nn.Module:
    model: nn.Module
    if pretrained:
        model = vgg16_bn(weights=VGG16_BN_Weights, progress=True)
    else:
        model = vgg16_bn()
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, 2)
    return model.to(device)

class MyCustomModel(nn.Module):
    def __init__(self, in_features: int) -> None:
        pass

def get_custom() -> nn.Module:
    return MyCustomModel(2).to(device)

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