#main code
# coding: utf-8
#pytorch=0.3.0-py36cuda8.0cudnn6.0_0 defaults
#torchvision=0.2.0
#if the version is not correct, it will raise error about "transform", googleNet pth error.


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
# import cv2
# get_ipython().run_line_magic('matplotlib', 'inline')
params = {
    'font.family': 'sans-serif',
    'axes.titlesize': 25,
    'axes.titlepad': 10,
}
pylab.rcParams.update(params)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


model_methods = [

    # ['resnet50', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

    # ['resnet152', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

    # ['resnet101', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

    # ['vgg16', 'cbpP5_C43_C33_C22_in', ['Pool5', 'Conv4_3', 'Conv3_3', 'Conv2_2', 'Pixel space'], 'camshow'], #test
    ['vgg16', 'cbpP5_C43_C33_C22_in', ['Pool5', 'Conv4_3', 'Conv3_3', 'Pixel space'], 'camshow'],  # test

    # ['vgg16','ebpfm30_29_27_15_in',['cbpfm30','cbpfm29','cbpfm27','cbpfm15', 'cbpfmin'],'camshow'],
    ## ['vgg16', 'ebpfm30_29_27_15_in', ['conv5_3', 'conv5_2', 'conv5_1', 'conv4_3', 'conv4_2'], 'camshow'],
    # ['vgg16_bn', 'ebpfm30_29_27_15_in', ['ebpfm30', 'ebpf29', 'ebpfm27', 'ebpfm15', 'ebpfmin'], 'camshow'],
    # ['vgg19', 'ebpfm36_33_27_15_in', ['ebpfm30', 'ebpf29', 'ebpfm27', 'ebpfm15', 'ebpfmin'], 'camshow'],
    # ['vgg13', 'ebpfm23_20_15_6_in', ['ebpfm30', 'ebpf29', 'ebpfm27', 'ebpfm15', 'ebpfmin'], 'camshow'],

    # ['resnext101_32x8d', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

    # ['mobilenet_v2', 'lrpMBv2', ['fm18', 'fm17', 'fm8', 'fm1', 'in'], 'camshow'],
    # #['mobilenet_v2', 'lrpMBv2', ['fm18', 'fm7', 'fm6', 'fm1', 'in'], 'camshow'],

    # ['googlenet', 'ebpGNly4_3_2_1_in', ['incep5b','incep4e','incep3b','maxpool2','inp'], 'camshow'],
    #
    # ['densenet121', 'ebpDB4_3_2_1_in', ['cbpDB4', 'cbpDB3', 'cbpDB2', 'cbpDB1', 'cbpDBin'], 'camshow'],

    # extract the model; visualizing method (layers for visualization in create_explainer.py);
    # names in the plotting; method for plotting

]



imgname='images/cat_dog'#cat_dog33'#2007_000733'#cat_dog'#

image_path = imgname+'.png'
# image_path = imgname+'.JPG'#'.JPEG'
image_class = None #101 # tusker
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
index=282

all_saliency_maps = []
all_index=[]
for model_name, method_name, _, _ in model_methods:  #model_name, method_name, _, _
    print(model_name,method_name)

    # import torchvision.transforms as transforms

    transf = get_preprocess(model_name, method_name) #indluding preprocessing the image differently according to models
    inp = transf(raw_img)#*0 #[3,224,224]

    inp = torch.autograd.Variable(inp.unsqueeze(0), requires_grad=True)

    model = utils.load_model(model_name)

    explainer = get_explainer(model, method_name,top_n) #key
    # target = torch.LongTensor([image_class]).cuda()
    saliency,index = explainer.explain(inp,index) #key #(inp, target)

    all_saliency_maps.append(saliency)#.cpu().numpy())

saveName='images/cat_dog'+'_'+model_methods[0][0]+'_'+str(index)+'_lrp_fcCWWW_convW+_absIn.png'


# imshow

fig=plt.figure(figsize=(12, 4))

for i, (saliency_maps,(model_name,method_name, method_fm_names,show_style)) in enumerate(zip(all_saliency_maps,model_methods)):
    for j in range(len(saliency_maps)):
# for j, (saliency_maps,(model_name,method_name, show_style)) in enumerate(zip(all_saliency_maps,model_methods)):

        plt.subplot(1, 4, j + 1)#(1, 6, j + 2) 

        saliency=saliency_maps[j]
        # saliency=saliency_maps
        saliency = saliency.sum(1).data.cpu().numpy()
        # saliency=saliency.abs().max(dim=1)[0].data.cpu().numpy()#for GradBP
        # saliency=saliency.sum(1).clamp_(0,1).cpu().numpy()
        # saliency[saliency<0]=0 #
        a = saliency.squeeze()
        # a = np.transpose(a, (1, 2, 0))

        method_fm_name=method_fm_names[j]

        if show_style == 'camshow':
            # viz.plot_cam(np.abs(saliency).max(axis=1).squeeze(), raw_img, 'jet', alpha=0.5)

            #binary
            # a[a > 0] = 1
            # a[a < 0] = 0
            ###
            viz.plot_cam(a, raw_img, 'jet', alpha=0.5)

        # labelname=labelnamedict[np.int(index.squeeze())]  #squeeze()!
        labelname=labelnamedict[np.int(index)]  #squeeze()!

        plt.axis('off')
        plt.title('%s' % (method_fm_name), fontsize=26)  


plt.tight_layout()
plt.imshow(attr, cmap=cmap)
plt.show()


