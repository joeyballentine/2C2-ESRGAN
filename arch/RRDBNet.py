import math

import torch
import torch.nn as nn

from . import block as B


class RRDBNet(nn.Module):
    def __init__(
        self,
        in_nc,
        out_nc,
        nf,
        nb,
        upscale=4,
        act_type="leakyrelu",
    ):
        super(RRDBNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))

        fea_conv = B.conv_block(in_nc, nf, act_type=None)
        rb_blocks = [
            RRDB(
                nf,
                act_type=act_type,
            )
            for _ in range(nb)
        ]
        LR_conv = B.conv_block(
            nf,
            nf,
            act_type=None,
        )

        upsampler = [
            B.upconv_block(nf, nf, act_type=act_type) for _ in range(n_upscale)
        ]
        HR_conv0 = B.conv_block(nf, nf, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, act_type=None)

        self.model = B.sequential(
            fea_conv,
            B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),
            *upsampler,
            HR_conv0,
            HR_conv1,
        )

    def forward(self, x, outm=None):
        x = self.model(x)

        if (
            outm == "scaltanh"
        ):  # limit output range to [-1,1] range with tanh and rescale to [0,1] Idea from: https://github.com/goldhuang/SRGAN-PyTorch/blob/master/model.py
            return (torch.tanh(x) + 1.0) / 2.0
        elif outm == "tanh":  # limit output to [-1,1] range
            return torch.tanh(x)
        elif outm == "sigmoid":  # limit output to [0,1] range
            return torch.sigmoid(x)
        elif outm == "clamp":
            return torch.clamp(x, min=0.0, max=1.0)
        else:  # Default, no cap for the output
            return x


class RRDB(nn.Module):
    """
    Residual in Residual Dense Block
    (ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks)
    """

    def __init__(
        self,
        nf,
        gc=32,
        act_type="leakyrelu",
    ):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )
        self.RDB2 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )
        self.RDB3 = ResidualDenseBlock_5C(
            nf,
            gc,
            act_type,
        )

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class ResidualDenseBlock_5C(nn.Module):
    def __init__(
        self,
        nf=64,
        gc=32,
        act_type="leakyrelu",
    ):
        super(ResidualDenseBlock_5C, self).__init__()

        self.conv1 = B.conv_block(
            nf,
            gc,
            act_type=act_type,
        )
        self.conv2 = B.conv_block(
            nf + gc,
            gc,
            act_type=act_type,
        )
        self.conv3 = B.conv_block(
            nf + 2 * gc,
            gc,
            act_type=act_type,
        )
        self.conv4 = B.conv_block(
            nf + 3 * gc,
            gc,
            act_type=act_type,
        )
        last_act = None
        self.conv5 = B.conv_block(
            nf + 4 * gc,
            nf,
            act_type=last_act,
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(torch.cat((x, x1), 1))
        if self.conv1x1:
            x2 = x2 + self.conv1x1(x)  # +
        x3 = self.conv3(torch.cat((x, x1, x2), 1))
        x4 = self.conv4(torch.cat((x, x1, x2, x3), 1))
        if self.conv1x1:
            x4 = x4 + x2  # +
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        if self.noise:
            return self.noise(x5.mul(0.2) + x)
        else:
            return x5 * 0.2 + x
