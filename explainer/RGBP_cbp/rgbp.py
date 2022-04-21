import types
from explainer.RGBP_cbp.functionsGBP import *


def get_layer(model, key_list):
    a = model
    for key in key_list:
        a = a._modules[key] #
    return a

class ExcitationBackpropExplainer(object):
    def __init__(self, model, top_n=None):
        self.model = model
        self.top_n=top_n
        self._override_backward()

    def _override_backward(self): #key
        def new_avgpool2d(self, x):
            return AVGPoolG(self,x) #useful for Densenet, otherwise no need
        def new_conv2d(self, x):
            return WoneConv2dG.apply(x, self.weight, self.bias, self.stride, self.padding,self.dilation,self.groups)
        def new_bn(self, x):
            return LRPBNG.apply(x, self.running_mean, self.running_var, self.weight, self.bias,
                               self.training)
        def replace(m):
            name = m.__class__.__name__
            if  name == 'Conv2d':#None
                m.forward = types.MethodType(new_conv2d, m)
            elif name=='BatchNorm2d':#None
                m.forward=types.MethodType(new_bn,m)
            elif name == 'AvgPool2d':   #None#'AvgPool2d':
                m.forward = types.MethodType(new_avgpool2d, m)
            #     # useful for Densenet, otherwise no need
        self.model.apply(replace) #iterate automatically for each module

        def new_linear3(self, x):
            return CLinearG.apply(x, self.weight, self.bias)
        if list(self.model.children())[-1].__class__.__name__=='Sequential': #for VGG
            m=list(self.model.children())[-1][-1]
        else: m=list(self.model.children())[-1]
        m.forward = types.MethodType(new_linear3, m)


    def explain(self, inp, ind=None):

        self.intermediate_act=[] #
        self.intermediate_vars = [] #

        output = self.model(inp)

        #set output object

        # ind=282
        if ind is None:
            topn=torch.topk(output,5)[1] #(output,20)[1]#
            ind=topn[0,self.top_n].data.clone()#top2[0,1]
            # ind2=topn[0,0].data.clone()#top2[0,1]
            ind=ind.cpu().numpy()
            print('pred class:',F.softmax(output,dim=1).cpu().detach().numpy()[0,topn[0,0]])

        grad_out = output.data.clone() #old version pytorch, change Variable to Tensor
        grad_out.fill_(0)
        # ind=243
        grad_out[0,ind]=1.0

        self.model.zero_grad()

        output.backward(grad_out, retain_graph=True)
        grad_inp=inp.grad*inp
        attmap_var = grad_inp.sum(1, keepdim=True)

        return attmap_var,ind

