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


#LRP以外方法，注意把resnet内部skip改回去！！
model_methods = [

    # ['resnet50', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

    # ['resnet18', 'lrply4_3_2_1_in', ['lrply4', 'lrply3', 'lrply2', 'lrply1', 'lrplyin'], 'camshow'],

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

    # ['shufflenet_v2_x1_0', 'lrpSFv2', ['conv5', 'stage3', 'stage2', 'conv1', 'in'], 'camshow'],

    # ['inception_v3', 'ebpGV3ly4_3_2_1_in', ['M7c','M6a','M5b','C4a','inp'], 'camshow'],
    #
    # ['googlenet', 'ebpGNly4_3_2_1_in', ['incep5b','incep4e','incep3b','maxpool2','inp'], 'camshow'],
    #
    # ['densenet121', 'ebpDB4_3_2_1_in', ['cbpDB4', 'cbpDB3', 'cbpDB2', 'cbpDB1', 'cbpDBin'], 'camshow'],

    # ['alexnet','ebpfm12_7_5_2_in',['cbpfm12','cbpfm7','cbpfm5','cbpfm2', 'cbpfmin'],'camshow'],

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
    # #ablation test
    #先把transf里的transforms.Normalize取消，然后置0，再在外面实现transforms.Normalize
    # transf2=transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
    # inp=transf2(inp)#*0

    ####
    # # for Padding+Resize
    # # padding
    # img=raw_img.copy()
    # if img.width < img.height:
    #     pad_w1 = int(round((img.height - img.width) / 2))  # round 四舍六入，五偶舍奇入
    #     pad_w2 = img.height - img.width - pad_w1
    #     pad_h1 = 0
    #     pad_h2 = 0
    # else:
    #     pad_w1 = 0
    #     pad_w2 = 0
    #     pad_h1 = int(round((img.width - img.height) / 2))  # round 四舍六入，五偶舍奇入
    #     pad_h2 = img.width - img.height - pad_h1
    #
    # transf3 = transforms.Compose([
    #     transforms.Pad(
    #         (pad_w1, pad_h1, pad_w2, pad_h2)),
    #     transforms.Resize((500, 500)),  # PIL.resize((256,256))
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225])
    # ])
    # inp=transf3(raw_img)

    # inp = utils.cuda_var(inp.unsqueeze(0), requires_grad=True) #very slow
    inp = torch.autograd.Variable(inp.unsqueeze(0), requires_grad=True)

    model = utils.load_model(model_name)
    # model.cuda()  #very slow

    explainer = get_explainer(model, method_name,top_n) #key
    # target = torch.LongTensor([image_class]).cuda()
    saliency,index = explainer.explain(inp,index) #key #(inp, target)
    # saliency,index = explainer.explain(inp,282) #key #(inp, target)
    # exit()

    # saliency = utils.upsample(saliency, (raw_img.height, raw_img.width))  #[1,1,224,224]

    # saliency=torch.Tensor(saliency)
    # saliency = saliency.sum(2) #error
    # saliency[saliency<0]=0

    all_saliency_maps.append(saliency)#.cpu().numpy())


    # #plot test,不同于下面的viz.plot_cam
    # a=saliency[4].cpu().numpy().sum(1).squeeze()
    # # a=cv2.resize(a,( raw_img.width,raw_img.height)) #注意cv2里宽和高的顺序！
    # a[a<0]=0
    # a /= np.max(np.abs(a))
    #
    # # # plt.imshow(raw_img)
    # # # plt.imshow(a, alpha=0.5, cmap="jet")
    # plt.imshow(a, cmap="seismic", clim=(-1, 1))  # clim 颜色渲染
    # saveName = imgname + '_' + model_methods[0][0] + '_' + str(index) + '_GBP.png'
    # # plt.show()
    # #plt.savefig(saveName)
    ####segmentation
    # a=a[:,:,np.newaxis]
    # y = raw_img * (a > 0.01)
    # plt.figure(figsize=(2.24, 2.24))
    # plt.imshow(y)
    ################
    # a=saliency[4].cpu().numpy().sum(1).squeeze();a[a<0]=0
    # a /= np.max(np.abs(a))
    # plt.imshow(a, cmap="seismic", clim=(-1, 1))  # clim 颜色渲染
    # plt.axis('off')
    # plt.gcf().set_size_inches(224 / 100, 224 / 100)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.show()
    # plt.savefig('images/cat_dog_lrp/fcCWWW282(a>0.01).png')
    # plt.savefig(imgname + '_' + model_methods[0][0] + '_207_CBP.png')

# saveName=imgname+'_'+model_methods[0][0]+'_'+str(index)+'_clrp_CWCWCW_convW+.png'
# saveName=imgname+'_'+model_methods[0][0]+'_'+str(index)+'_clrp_CWW+W+_convW+.png'
# saveName='images/cat_dog_lrp/fcCWWW/fcCWWW282(a>0.01)'+'_'+model_methods[0][0]+'_'+str(index)+'_lrp_fcCWWW.png'
# saveName='images/cat_dog_lrp/fcW-WW/cat_dog'+'_'+model_methods[0][0]+'_'+str(index)+'_lrp_fcW-WW.png'

saveName='images/cat_dog_lrp/fcCWWW/cat_dog'+'_'+model_methods[0][0]+'_'+str(index)+'_lrp_fcCWWW_convW+_absIn.png'


# imshow

# plt.figure(figsize=(25, 15))
# plt.subplot(3, 5, 1)
fig=plt.figure(figsize=(12, 4))
# plt.figure(figsize=(8, 4)) #pixel X10
# plt.subplot(1, 3, 1)
# plt.figure(figsize=(6.6,2),dpi=100)#(16, 4)) #pixel x10

# plt.subplot(1, 6, 1)
# plt.imshow(raw_img) #raw_img是预处理之前的图像，并不是真正输入网络的图像
# # a=np.transpose(inp[0].data.numpy(), (1, 2, 0))
# # plt.imshow(a) #真正输入网络的图像
# plt.axis('off')
# # plt.title('model:\n'+model_methods[0][1], fontsize=12)#class_id=101 (Tusker)
# plt.title('Tiger cat', fontsize=16)

for i, (saliency_maps,(model_name,method_name, method_fm_names,show_style)) in enumerate(zip(all_saliency_maps,model_methods)):
    for j in range(len(saliency_maps)):
# for j, (saliency_maps,(model_name,method_name, show_style)) in enumerate(zip(all_saliency_maps,model_methods)):
        # plt.subplot(3, 5, i + 2 + i // 4)
        # plt.subplot(1, 3, i + 2)
        plt.subplot(1, 4, j + 1)#(1, 6, j + 2) #j从0开始，第一幅图是原图，所以j+2从2开始

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
        plt.title('%s' % (method_fm_name), fontsize=26)  #16# 有时候文字太长，导致保存的图像不完整，或倒置
        # plt.title('%s\n%s' % (labelname,method_fm_name),fontsize=12) #有时候文字太长，导致保存的图像不完整，或倒置 #
        # plt.title('%s\n%s' % (labelname,method_name),fontsize=16) #有时候文字太长，导致保存的图像不完整，或倒置



plt.tight_layout()
# plt.savefig(imgname+'_ebp_w_no_offset.png')
# plt.savefig(imgname + '_' + model_methods[0][0] +'_' +str(index)+ '_layers_GradBP_0514_2.png',bbox_inches='tight')
# plt.savefig(imgname + '_' + model_methods[0][0] +'_' +str(index)+ '_layers_0514_ExitNeg.png',bbox_inches='tight')
# plt.savefig('images/change_Beta/cat_dog' + '_' + model_methods[0][0] +'_' +str(index)+ '_layers_TSG_Beta1.png',bbox_inches='tight')
###### plt.imshow(attr, cmap=cmap, clim=(-0.3,1.05))
plt.show()
# plt.ioff()
# plt.clf()
# plt.close()
# exit()

# from matplotlib.backends.backend_pdf import PdfPages
# #
# with PdfPages("cat_dog_vgg16_layers_282_TSG_20210409.pdf") as pdf:
# # with PdfPages("cat_dog_vgg16_layers_282_BPxFMpostSum_20210409.pdf") as pdf:
#
#     params = {
#         'figure.figsize': '8, 5'
#     }
#     plt.rcParams.update(params)
#
#     # plt.rcParams['font.sans-serif'] = ['FangSong']
#
#     pdf.savefig(bbox_inches='tight')

