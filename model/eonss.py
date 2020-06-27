import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GdnFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, x, gamma, beta):
        # save variables for backprop
        ctx.save_for_backward(x, gamma, beta)
        n, c, h, w = list(x.size())
        # input is formatted as NCHW, here we need it to be NHWC
        tx = x.permute(0, 2, 3, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        # rbeta = beta.repeat(n * h * w, 1)
        denominator = tx2.mm(gamma) + beta
        ty = tx / torch.sqrt(denominator)
        y = ty.view(n, h, w, c)
        y = y.permute(0, 3, 1, 2).contiguous()
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        x, gamma, beta = ctx.saved_variables
        # input is formatted as NCHW, here we need it to be NHWC
        n, c, h, w = list(grad_output.size())
        tx = x.permute(0, 2, 3, 1).contiguous()
        tx = tx.view(-1, c)
        tx2 = tx * tx
        # rbeta = beta.repeat(n * h * w, 1)
        denominator = tx2.mm(gamma) + beta

        tdzdy = grad_output.permute(0, 2, 3, 1).contiguous()
        tdzdy = tdzdy.view(-1, c)
        gy = (tdzdy * torch.pow(denominator, -0.5) - (tdzdy * tx *
              torch.pow(denominator, -1.5)).mm(gamma.t()) * tx)
        gy = gy.view(n, h, w, c)
        grad_input = gy.permute(0, 3, 1, 2).contiguous()
        tmp = -0.5 * torch.pow(denominator, -1.5) * tx * tdzdy
        grad_beta = torch.sum(tmp, 0)
        grad_gamma = tx2.t().mm(tmp)
        return grad_input, grad_gamma, grad_beta


class Gdn(nn.Module):
    def __init__(self, input_channel):
        super(Gdn, self).__init__()
        self.input_channel = input_channel
        # nn.Parameter is a special kind of Variable, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters can never be volatile and, different than Variables,
        # they require gradients by default.
        self.gamma = nn.Parameter(torch.Tensor(input_channel, input_channel))
        self.beta = nn.Parameter(torch.Tensor(input_channel))

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return GdnFunction.apply(input, self.gamma, self.beta)

    def __str__(self):
        return self.__class__.__name__ + '(gamma_size=(%d, %d), beta_size=(%d))' %(self.gamma.size()[0], self.gamma.size()[1], self.beta.size()[0])

    __repr__ = __str__


class EONSS(nn.Module):
    def __init__(self):
        super(EONSS, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 5, stride=2, padding=2)
        self.gdn1 = Gdn(8)
        self.conv2 = nn.Conv2d(8, 16, 5, stride=2, padding=2)
        self.gdn2 = Gdn(16)
        self.conv3 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.gdn3 = Gdn(32)
        self.conv4 = nn.Conv2d(32, 64, 3, stride=1, padding=0)
        self.gdn4 = Gdn(64)

        self.st2_fc1 = nn.Conv2d(64, 256, 1, stride=1, padding=0)
        self.st2_gdn1 = Gdn(256)
        self.st2_fc2 = nn.Conv2d(256, 1, 1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, Gdn):
                m.gamma.data.fill_(1)
                m.beta.data.fill_(1e-2)

    def forward(self, x):
        batch_size = x.size()[0]

        x = F.max_pool2d(self.gdn1(self.conv1(x)), (2, 2))
        x = F.max_pool2d(self.gdn2(self.conv2(x)), (2, 2))
        x = F.max_pool2d(self.gdn3(self.conv3(x)), (2, 2))
        x = F.max_pool2d(self.gdn4(self.conv4(x)), (2, 2))

        y2 = self.st2_gdn1(self.st2_fc1(x))
        s = self.st2_fc2(y2)
        s = s.view(batch_size, -1)

        return s

    # gdn parameter post-processing
    def _gdn_param_proc(self):
        for m in self.modules():
            if isinstance(m, Gdn):
                m.beta.data.clamp_(min=2e-10)
                m.gamma.data.clamp_(min=2e-10)
                m.gamma.data = (m.gamma.data + m.gamma.data.t()) / 2
