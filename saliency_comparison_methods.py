
import torch
# import torchvision.models as m
# vgg=m.vgg16(pretrained=True) #test
# torch.nn.AdaptiveAvgPool2d
import sys
sys.path.append('./')

from create_explainer import get_explainer
from preprocess import get_preprocess
import utils
import viz
import time
import os
import pylab
import numpy as np
import matplotlib.pyplot as plt

params = {
    'font.family': 'sans-serif',
    'axes.titlesize': 25,
    'axes.titlepad': 10,
}
pylab.rcParams.update(params)

mm='vgg16'
#'shufflenet_v2_x0_5'#'mobilenet_v2'
#'densenet121'
#'googlenet'#'inception_v3'
#'resnext101_32x8d'#'resnet152'#'resnet101'#'resnet50'#
# 'vgg16_bn'#'vgg16'#

model_methods = [
    [mm, 'guided_backprop',    'camshow'],
    [mm, 'TSGB', 'camshow'],
]

imgname='images/cat_dog'
image_path = imgname+'.png'
# image_path = imgname+'.JPEG' #'.jpg'#'.JPG'#'.JPEG'
# image_class = None #101 # tusker
raw_img = viz.pil_loader(image_path)
# plt.figure(figsize=(5, 5))
# plt.imshow(raw_img)
# plt.axis('off')
# plt.title('resnet50')#class_id=101 (Tusker)
# plt.show()

with open ('imagenet1000_clsidx_to_labels.txt',"r") as f:
    labelnamedict=eval(f.read()) #eval() transform string to dict


#explain and get saliency maps

top_n=0#output node--top_n=9, means top10. index != top_n
index=None#282#23#None#717

all_saliency_maps = []
all_index=[]
for model_name, method_name, _ in model_methods:  #model_name, method_name, _, _
    print(model_name,method_name)


    transf = get_preprocess(model_name, method_name) #indluding preprocessing the image differently according to models
    inp = transf(raw_img)#*0 #[3,224,224]


    # inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True) #very slow
    inp = torch.autograd.Variable(inp.unsqueeze(0), requires_grad=True)

    model = utils.load_model(model_name)

    #########
    explainer = get_explainer(model, method_name,top_n) #key
    # target = torch.LongTensor([image_class]).cuda()
    saliency,index = explainer.explain(inp,index) #key #(inp, target)
    # saliency,index = explainer.explain(inp,282) #key #(inp, target)

    saliency =torch.nn.functional.interpolate(saliency,(inp.shape[2], inp.shape[3]),mode='bilinear', align_corners=False).squeeze()  #[n,1,w,h]->[n,224,224]

    all_saliency_maps.append(saliency)#.cpu().numpy()) #[n,c,224,224]


# saveName='images/examles/cat_dog'+'_'+model_methods[0][0]+'_'+str(index)+'_result.png'

# imshow

# plt.figure(figsize=(25, 10))#大图
# plt.subplot(3, 5, 1)
plt.figure(figsize=(12, 4)) #pixel X10 适合保存一行3个图，紧密布局
# plt.figure(figsize=(6.6,2),dpi=100)#(16, 4)) #pixel x10
plt.subplot(1, 3, 1)
# plt.subplot(1, 7, 1)#(1, 8, 1)#(1, 7, 1)
plt.imshow(raw_img.resize((224,224))) #(224)) # raw_img是预处理之前的图像，并不是真正输入网络的图像
plt.axis('off')


for j, (saliency_maps,(model_name,method_name, show_style)) in enumerate(zip(all_saliency_maps,model_methods)):
        # plt.subplot(3, 5, j + 2 + j // 4)
        plt.subplot(1, 3, j + 2)
        # plt.subplot(1, 8, j + 2)  # j从0开始，第一幅图是原图，所以j+2从2开始
        # plt.subplot(1, 7, j + 2) #j从0开始，第一幅图是原图，所以j+2从2开始

        a = saliency_maps.squeeze().data
        a = a.numpy()
        if show_style == 'camshow':
            # viz.plot_cam(np.abs(saliency).max(axis=1).squeeze(), raw_img, 'jet', alpha=0.5)

            #binary
            # a[a > 0] = 1
            a[a < 0] = 0
            ###
            viz.plot_cam(a,  'clear_plot')


        labelname=labelnamedict[np.int(index)]  #squeeze()!

        plt.axis('off')

plt.tight_layout()

plt.show()

print(str(index)+ labelname)
print("Done!")





