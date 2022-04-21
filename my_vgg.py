import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(False),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights: #即使init_weights=False，初始化的时候也会在父类nn.Module里初始化
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        # #求fm30grad
        # x = torch.autograd.Variable(x, requires_grad=True) #输入、输出必须为相同的x，否则无法求梯度！
        # x2 = x.view(x.size(0), -1)
        # x3 = self.classifier(x2)

        # x=torch.autograd.Variable(x,requires_grad=True)
        # # my FCmap
        # def fcmapmul(map,w): #map[n,c1,h,w], w[c,c1,h,w]
        #     fmap=torch.zeros(map.shape[0],w.shape[0],map.shape[2],map.shape[3])
        #     for n in range(map.shape[0]):
        #             fm=map[n]*w #such as [4096,512,7,7]
        #             fmap[n]=fm.sum(1)
        #     # for n in range(map.shape[0]):
        #     #     for c in range(w.shape[0]):
        #     #         fm=map[n]*w[c] #such as [512,7,7]
        #     #         fmap[n,c]=fm.sum(0)
        #     return fmap #[n,c,h,w]
        #
        # wfc0=self._modules['classifier']._modules['0'].weight
        # wfc0_reshape=wfc0.view(-1,512,7,7)
        # bfc0=self._modules['classifier']._modules['0'].bias
        # fc0_z= fcmapmul(x,wfc0_reshape) #[n,4096,7,7]
        # fc0_out=fc0_z.sum((2,3))+bfc0
        # fc0=fc0_z+bfc0.unsqueeze(-1).unsqueeze(-1)/49
        # fc1=fc0*(fc0_out.unsqueeze(-1).unsqueeze(-1)>0).float() #ReLU(fc0)
        #
        # wfc3=self._modules['classifier']._modules['3'].weight
        # wfc3_reshape=wfc3.unsqueeze(-1).unsqueeze(-1) #[4096,4096,1,1]
        # bfc3=self._modules['classifier']._modules['3'].bias
        # fc3_z= fcmapmul(fc1,wfc3_reshape) #[n,4096,7,7]
        # fc3_out=fc3_z.sum((2,3))+bfc3
        # fc3=fc3_z+bfc3.unsqueeze(-1).unsqueeze(-1)/49
        # fc4=fc3*(fc3_out.unsqueeze(-1).unsqueeze(-1)>0).float()
        #
        # wfc6=self._modules['classifier']._modules['6'].weight
        # wfc6_reshape=wfc6.unsqueeze(-1).unsqueeze(-1) #[1000,4096,1,1]
        # bfc6=self._modules['classifier']._modules['6'].bias
        # fc6_z= fcmapmul(fc4,wfc6_reshape) #[n,1000,7,7]
        # fc6_out=fc6_z.sum((2,3))+bfc6
        # fc6=fc6_z+bfc6.unsqueeze(-1).unsqueeze(-1)/49
        # # out=fc6*(fc6_out.unsqueeze(-1).unsqueeze(-1)>0).float()
        #
        # import matplotlib.pyplot as plt
        # import numpy as np

        # a = fc6[0, 282].data.squeeze().numpy()
        # a /= (np.abs(a).max()+1e-10)
        # plt.imshow(a, cmap="seismic", clim=(-1, 1))

        # a = fc6.data.sum(1).squeeze().numpy()
        # a /= (np.abs(a).max()+1e-10)
        # plt.imshow(a, cmap="seismic", clim=(-1, 1))

        # return fc6_out#x
        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16']))
    return model


def vgg16_bn(pretrained=False, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16_bn']))
    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model
