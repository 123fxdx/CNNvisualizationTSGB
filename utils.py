from torchvision import models
from torch.autograd import Variable
# from torch._thnn import type2backend
import my_resnet
import my_vgg
import my_densenet
import my_resnet_pytorch1_4
import my_googlenet_pytorch1_4
import my_mobilenetv2_pytorch1_4

def load_model(arch):
    '''
    Args:
        arch: (string) valid torchvision model name,
            recommendations 'vgg16' | 'googlenet' | 'resnet50'
    '''

    if 'resn' in arch: #include 'resnet' and 'resnext'
        # model=my_resnet.__dict__[arch](pretrained=True)
        model=my_resnet_pytorch1_4.__dict__[arch](pretrained=True) #also OK

    elif arch == 'googlenet':
    #     # from googlenet import get_googlenet
    #     # model = get_googlenet(pretrain=True)
        model=my_googlenet_pytorch1_4.__dict__[arch](pretrained=True)

    elif 'densenet' in arch:
        model = my_densenet.__dict__[arch](pretrained=True)
    else:
        model = models.__dict__[arch](pretrained=True)#False)#True)#

    model.eval()
    return model

def cuda_var(tensor, requires_grad=False):
    return Variable(tensor.cuda(), requires_grad=requires_grad)
