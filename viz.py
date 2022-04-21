import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
# import cv2


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def plot_cam(attr, cmap='clear_plot'):
    #my plot
    attr/=(np.abs(attr).max()+1e-10)

    ###############
    import matplotlib.colors as col
    import matplotlib.cm as cm

    ######################################### 0514
    color=cm.get_cmap('jet')
    ##### balance
    newcolor=color(np.linspace(0.25,0.95,40)) #(0.2,0.9,20)#
    newcolor[0:1,:]=np.array([0.1,0.1,0.75,1]) #

    newcolor0=color(np.linspace(0.2,1,40)) #(0.2,0.9,20)#
    newcolor[1:2, :] =newcolor0[0,:]#np.array([0.3,0.3,1,1])#newcolor[1 ,:]
    ####


    clear_plot = col.LinearSegmentedColormap.from_list('own4', newcolor)
    #################################

    #############################
    if cmap=='clear_plot':
        plt.imshow(attr, cmap=clear_plot) #use this
    else:
        plt.imshow(attr, cmap='seismic',clim=(-1, 1))




def plot_bbox(bboxes, xi, linewidth=1):
    ax = plt.gca()
    ax.imshow(xi)

    if not isinstance(bboxes[0], list):
        bboxes = [bboxes]

    for bbox in bboxes:
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=linewidth, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')

