
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torch

class CLinearG(Function):

    @staticmethod
    def forward(ctx, inp, weight, bias=None):
        ctx.save_for_backward(inp, weight, bias)
        output = inp.matmul(weight.t())
        if bias is not None:
            output += bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inp, weight, bias = ctx.saved_variables

        w_pos = weight.clone().clamp(min=0);
        output_pos = inp.matmul(w_pos.t())
        w_neg = weight.clone().clamp(max=0);
        output_neg = inp.matmul(w_neg.t())
        beta = (output_pos )/ output_neg.abs()
        Beta = 1.0 * beta
        # for Resnet 0.75 (0.8, 0.85, 0.9)
        gw_pos = grad_output.matmul(w_pos)
        gw_neg=(grad_output*Beta).matmul(w_neg)
        gw=gw_pos+gw_neg

        return gw, None, None


class LRPBNG(Function): #also as LRP+BN

    @staticmethod
    def forward(ctx, inp, running_mean, running_var, weight, bias, #forward里都是tensor
               training):

        output=F.batch_norm(inp, running_mean, running_var, weight=weight, bias=bias,
               training=training)
        # ctx.save_for_backward(inp,running_mean, running_var, weight, bias,output)
        ctx.save_for_backward(inp,running_mean, running_var, weight, bias)
        #cancel the inplace operation to save "output".
        #### ctx.output=output #the saved variable may be modified by Inplace operation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # inp, running_mean, running_var, weight, bias,output=ctx.saved_variables
        inp, running_mean, running_var, weight, bias=ctx.saved_variables
        # output=ctx.output
        output=F.batch_norm(inp, running_mean, running_var, weight=weight, bias=bias,
               training=False) #test

        grad_input=grad_output*output/(inp+1e-10)

        return grad_input, None, None, None, None, None, None

class WoneConv2dG(Function):

    @staticmethod
    def forward(ctx, inp, weight, bias, stride, padding,dilation,groups):

        # ctx.save_for_backward(inp, weight, bias,output)
        ctx.save_for_backward(inp, weight, bias)
        ctx.padding = padding
        ctx.stride =stride
        ctx.dilation = dilation
        ctx.groups = groups

        # ###for pytorch0.2
        inp=Variable(inp)
        weight=Variable(weight)
        if type(bias)==torch.Tensor:
            #if type(bias)!=None:#if bias is not None:
            ##if bias==None:#error! when bias has multi-dimension
            bias=Variable(bias)
        # ###

        output = F.conv2d(inp, weight, bias, stride=stride, padding=padding,dilation=dilation,groups=groups)  #

        # ctx.output=output#the saved variable may be modified by Inplace operation
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        # inp, weight, bias, output = ctx.saved_variables
        inp, weight, bias = ctx.saved_variables
        stride, padding, dilation, groups = ctx.stride, ctx.padding, ctx.dilation, ctx.groups
        # output=ctx.output

        output = F.conv2d(inp, weight, bias, stride=stride, padding=padding,dilation=dilation,groups=groups)  #

        c_output = grad_output* output#

        inp_norm=torch.abs(inp)
        outpadding1=inp.size(2)-((grad_output.size(2)-1)*stride[0]-2*padding[0]+dilation[0]*(weight.size(2)-1)+1) #!
        outpadding2=inp.size(3)-((grad_output.size(3)-1)*stride[1]-2*padding[1]+dilation[1]*(weight.size(3)-1)+1) #!

        ##### slim code for accelerate
        one_weight= torch.ones([1,1,weight.shape[2],weight.shape[3]]);#slim code
        in_sum=inp_norm.sum(1,True)#slim code
        out = F.conv2d(in_sum, one_weight, None, stride=stride,padding=padding,dilation=dilation,groups=1) #pos_bias##slim code
        c_output_sum=c_output.sum(1,True)#slim code
        normalized_c_output_sum=c_output_sum/(out + 1e-10)#slim code
        gw=F.conv_transpose2d(normalized_c_output_sum,one_weight,None,stride=stride,padding=padding,\
        dilation=dilation,groups=1,output_padding=(outpadding1,outpadding2))#slim code
        gw=gw.repeat(1,inp.size(1),1,1)#slim code
        ####

        ####if the above slim code doesn't work, use this code
        # one_weight= torch.ones_like(weight)
        # one_weight = Variable(one_weight)
        # out = F.conv2d(inp_norm, one_weight, None, stride=stride,padding=padding,dilation=dilation,groups=groups)
        # normalized_c_output=c_output/(out + 1e-10)
        # gw=F.conv_transpose2d(normalized_c_output,one_weight,None, stride=stride,padding=padding,dilation=dilation,groups=groups,output_padding=(outpadding1,outpadding2))
        ####

        gw=gw*inp.sign()

        return gw, None, None, None, None,None,None


class PreHook_AP(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_variables[0]
        # grad_input = input *grad_output
        #
        # grad_input=input*(input>0).float() * grad_output
        eps = 1e-10
        grad_input=grad_output/ (input + eps)#*input.sign()#*(input>0).float()#*input.abs()

        return grad_input
        # return grad_output #for Grad

class PostHook_AP(Function):  #

    @staticmethod
    def forward(ctx, out):
        ctx.save_for_backward(out)
        return out.clone()

    @staticmethod
    def backward(ctx, grad_output):
        out=ctx.saved_variables[0]
        eps = 1e-10
        # grad_input = grad_output/ (out + eps)
        #
        grad_input = grad_output*out#.sign()#.abs()#/ (out + eps) #out.clamp(0)#
        # grad_input[grad_input<0]=0
        return grad_input
        # return grad_output #for Grad

# AVG_Pool
def AVGPoolG(self, input): #key
    input= PreHook_AP.apply(input)
    output = F.avg_pool2d(input,self.kernel_size,self.stride,self.padding)
    output = PostHook_AP.apply(output)

    return output
