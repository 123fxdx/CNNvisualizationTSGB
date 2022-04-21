import torch
from explainer.backprop import VanillaGradExplainer

from torch.autograd import Function, Variable
import torch.nn.functional as F
import types
from torch.nn.modules.utils import _pair


def get_layer(model, key_list):
    a = model
    for key in key_list:
        a = a._modules[key]
    return a

class GradCAMExplainer(VanillaGradExplainer):
    def __init__(self, model, target_layer_name_keys=None, use_inp=False,top_n=None):
        self.model = model #
        super(GradCAMExplainer, self).__init__(self.model,top_n)#
        self.target_layer = get_layer(model, target_layer_name_keys)
        self.use_inp = use_inp
        self.intermediate_act = []
        self.intermediate_grad = []
        self._register_forward_backward_hook()
        self.top_n=top_n

    def _register_forward_backward_hook(self):
        def forward_hook_input(m, i, o):
            self.intermediate_act.append(i[0].data.clone()) #it is special for 'avgpool' layer to extract input features
        def forward_hook_output(m, i, o):
            self.intermediate_act.append(o.data.clone())


        def backward_hook_in(m, grad_i, grad_o): # my code, for input layer
            self.intermediate_grad.append(grad_i[0].data.clone())
            print('backward_hook_in shape',grad_i[0].shape)
        def backward_hook(m, grad_i, grad_o): #
            self.intermediate_grad.append(grad_o[0].data.clone())
            print('backward_hook shape',grad_o[0].shape)

        if self.use_inp: #FM和Grad的in,out要对应
            self.target_layer.register_forward_hook(forward_hook_input)
            self.target_layer.register_backward_hook(backward_hook_in)
        else:
            self.target_layer.register_forward_hook(forward_hook_output)
            self.target_layer.register_backward_hook(backward_hook)

        # self.target_layer.register_backward_hook(backward_hook)

    def _reset_intermediate_lists(self):
        self.intermediate_act = []
        self.intermediate_grad = []

    def explain(self, inp, ind=None):
        self._reset_intermediate_lists()

        _,ind = super(GradCAMExplainer, self)._backprop(inp, ind) #,top_n=self.top_n) #

        grad = self.intermediate_grad[0] #
        act = self.intermediate_act[0]

        weights = grad.sum((2,3),keepdim=True)#sum along spatial axes while keeping number of axes
        cam = weights * act
        cam = cam.sum(1,keepdim=True)
        cam = torch.clamp(cam, min=0)
        ##
        # cam=F.interpolate(cam,(inp.shape[2], inp.shape[3]),mode='bilinear', align_corners=False).squeeze()

        return cam,ind

