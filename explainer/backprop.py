import numpy as np
from torch.autograd import Variable, Function
import torch
import types


class VanillaGradExplainer(object):
    def __init__(self, model,top_n):
        self.model = model
        self.top_n = top_n

    def _backprop(self, inp, ind): #,top_n=None):
        output = self.model(inp)
        # y = self.model(inp)
        # output=torch.nn.Softmax(dim=1)(y)

        # self.top_n = top_n
        if ind is None:
            # ind = output.data.max(1)[1]
            topn=torch.topk(output,5)[1]# vgg16: 243,254,242,245,180,539,182,247,179,282
            ind=topn[0,self.top_n].data.clone()
        grad_out = output.data.clone()
        grad_out.fill_(0)
        grad_out[0,ind]=1.0
        output.backward(grad_out, retain_graph=True)
        return inp.grad.data,ind

    def explain(self, inp, ind=None):
        attmap_var,ind=self._backprop(inp, ind)
        attmap_var=attmap_var.abs().max(dim=1,keepdim=True)[0]
        return attmap_var,ind

# class VanillaGradExplainer(object): #dObjectLoss_dX
#     def __init__(self, model,top_n):
#         self.model = model
#         self.top_n = top_n
#
#     def _backprop(self, inp, ind): #,top_n=None):
#         output = self.model(inp)
#         output=torch.nn.Softmax(dim=1)(output)#
#         # self.top_n = top_n
#         if ind is None:
#             # ind = output.data.max(1)[1]
#             topn=torch.topk(output,30)[1]# vgg16: 243,254,242,245,180,539,182,247,179,282
#             ind=topn[0,self.top_n].data.clone()
#         grad_out = output.data.clone()
#         grad_out.fill_(0)
#         grad_out[0,ind]=1.0
#         # grad_out.scatter_(1, ind.unsqueeze(0).t(), 1.0)
#         loss=(output-grad_out).pow(2)#.sum() #
#         loss.backward(grad_out) #
#         # output.backward(grad_out)
#         return inp.grad.data,ind
#
#     def explain(self, inp, ind=None):
#         return self._backprop(inp, ind)


class GradxInputExplainer(VanillaGradExplainer):
    def __init__(self, model,top_n):
        super(GradxInputExplainer, self).__init__(model,top_n)

    def explain(self, inp, ind=None):
        grad,ind = self._backprop(inp, ind)
        attmap_var= inp.data * grad
        attmap_var=attmap_var.abs().max(dim=1,keepdim=True)[0]
        return attmap_var,ind

class SaliencyExplainer(VanillaGradExplainer):
    def __init__(self, model,top_n):
        super(SaliencyExplainer, self).__init__(model,top_n)

    def explain(self, inp, ind=None):
        grad,ind = self._backprop(inp, ind)
        return grad.abs(),ind


class IntegrateGradExplainer(VanillaGradExplainer):
    def __init__(self, model, steps=100):
        super(IntegrateGradExplainer, self).__init__(model)
        self.steps = steps

    def explain(self, inp, ind=None):
        grad = 0
        inp_data = inp.data.clone()

        for alpha in np.arange(1 / self.steps, 1.0, 1 / self.steps): #slow
            new_inp = Variable(inp_data * alpha, requires_grad=True)
            g,ind = self._backprop(new_inp, ind)
            grad += g

        return grad * inp_data / self.steps,ind


class DeconvExplainer(VanillaGradExplainer):
    def __init__(self, model):
        super(DeconvExplainer, self).__init__(model)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                grad_inp = torch.clamp(grad_output, min=0)
                return grad_inp

        def new_forward(self, x):
            return _ReLU.apply(x)

        def replace(m):
            if m.__class__.__name__ == 'ReLU':
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


class GuidedBackpropExplainer(VanillaGradExplainer):
    def __init__(self, model,top_n):
        super(GuidedBackpropExplainer, self).__init__(model,top_n)
        self._override_backward()

    def _override_backward(self):
        class _ReLU(Function):
            @staticmethod
            def forward(ctx, input):
                output = torch.clamp(input, min=0)
                ctx.save_for_backward(output)
                return output

            @staticmethod
            def backward(ctx, grad_output):
                output, = ctx.saved_tensors
                mask1 = (output > 0).float()
                mask2 = (grad_output.data > 0).float() #"Guided" relu
                grad_inp = mask1 * mask2 * grad_output.data
                grad_output.data.copy_(grad_inp)
                return grad_output

        def new_forward(self, x):
            return _ReLU.apply(x) #

        def replace(m):
            if m.__class__.__name__ == 'ReLU': #
                m.forward = types.MethodType(new_forward, m)

        self.model.apply(replace)


# modified from https://github.com/PAIR-code/saliency/blob/master/saliency/base.py#L80
class SmoothGradExplainer(object):
    def __init__(self, base_explainer, stdev_spread=0.15,
                nsamples=25, magnitude=True):
        self.base_explainer = base_explainer
        self.stdev_spread = stdev_spread
        self.nsamples = nsamples
        self.magnitude = magnitude

    def explain(self, inp, ind=None):
        stdev = self.stdev_spread * (inp.data.max() - inp.data.min())

        total_gradients = 0
        origin_inp_data = inp.data.clone()

        for i in range(self.nsamples):
            noise = torch.randn(inp.size()).cuda() * stdev
            inp.data.copy_(noise + origin_inp_data)
            grad,ind = self.base_explainer.explain(inp, ind)

            if self.magnitude:
                total_gradients += grad ** 2
            else:
                total_gradients += grad

        return total_gradients / self.nsamples,ind