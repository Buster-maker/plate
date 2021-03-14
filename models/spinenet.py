
import argparse
import yaml

from models.experimental import *
from models.common import Focus, BottleneckCSP, SPP
from models.yolo1 import Detect

# from alfred.dl.torch.common import device
from utils.grneral import check_anchor_order, make_divisible, check_file, set_logging
from utils.torch_utils import initialize_weights
import math
from torch import nn

from nb.torch.blocks.bottleneck_blocks import SimBottleneckCSP
from nb.torch.blocks.trans_blocks import Focus
from nb.torch.blocks.head_blocks import SPP
from nb.torch.blocks.conv_blocks import ConvBase
from nb.torch.backbones.spinenet import SpineNet

from nb.torch.utils import device


class YoloV6(nn.Module):

    def __init__(self, cfg, ch=3):
        super(YoloV6, self).__init__()

        with open(cfg) as f:
            self.md = yaml.load(f, Loader=yaml.FullLoader)
        self.nc = self.md['nc']
        self.anchors = self.md['anchors']
        self.na = len(self.anchors[0]) // 2  # number of anchors
        print('model num classes is: {}'.format(self.nc))
        print('model anchors is: {}'.format(self.anchors))

        # divid by, fixed currently due to SpineNet backbone
        cd = 2
        wd = 3

        self.backbone = SpineNet(arch='49S', output_level=[3, 4, 5])

        # PANet
        self.conv5 = ConvBase(256//cd, 256//cd)
        # self.conv5 = SimBottleneckCSP(1024//cd, 1024//cd, n=3//wd, shortcut=False)
        self.up1 = nn.Upsample(scale_factor=2)
        self.csp5 = SimBottleneckCSP(512//cd, 256//cd, n=3//wd, shortcut=False)

        self.conv6 = ConvBase(256//cd, 256//cd)
        self.up2 = nn.Upsample(scale_factor=2)
        self.csp6 = SimBottleneckCSP(512//cd, 256//cd, n=3//wd, shortcut=False)

        self.conv7 = ConvBase(256//cd, 256//cd, 3, 2)
        self.csp7 = SimBottleneckCSP(512//cd, 512//cd, n=3//wd, shortcut=False)

        self.conv8 = ConvBase(512//cd, 512//cd, 3, 2)
        self.csp8 = SimBottleneckCSP(512//cd, 1024//cd, n=3//wd, shortcut=False)

        self.detect = Detect(self.nc, self.anchors, [256//cd, 512//cd, 1024//cd])

        # forward to get Detect lay params dynamically
        s = 512  # 2x min stride
        self.detect.stride = torch.tensor(
            [s / x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])  # forward
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        self._initialize_biases()
        initialize_weights(self)

        # for compatible, in check_anchors in train.py
        self.model = [self.detect]

    def _initialize_biases(self, cf=None):
        m = self.detect  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s) ** 2)
            b[:, 5:] += math.log(0.6 / (m.nc - 0.99)
                                 ) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def _build_head(self, p3, p4, p5, feas):
        h_p5 = self.conv5(feas)  # head P5
        x = self.up1(h_p5)
        x_concat = torch.cat([x, p4], dim=1)
        x = self.csp5(x_concat)

        h_p4 = self.conv6(x)  # head P4
        x = self.up2(h_p4)
        x_concat = torch.cat([x, p3], dim=1)
        x_small = self.csp6(x_concat)

        x = self.conv7(x_small)
        x_concat = torch.cat([x, h_p4], dim=1)
        x_medium = self.csp7(x_concat)

        x = self.conv8(x_medium)
        x_concat = torch.cat([x, h_p5], dim=1)
        x_large = self.csp8(x)
        return x_small, x_medium, x_large

    def forward(self, x, augment=False):
        # we not using augment at all
        p3, p4, p5 = self.backbone(x)
        # print('p3 ', p3.shape)
        # print('p4 ', p4.shape)
        # print('p5 ', p5.shape)
        # print('feas ', feas.shape)
        xs, xm, xl = self._build_head(p3, p4, p5, p5)
        # print(xs.shape)
        # print(xm.shape)
        # print(xl.shape)
        x = self.detect([xs, xm, xl])
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str,
                        default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    # opt.cfg = check_file(opt.cfg)  # check file
    # Create model
    model = YoloV6(opt.cfg).to(device)
    # model.train()
    model.eval()
    # print(model)

    a = torch.randn([1, 3, 1280, 640, ]).to(device)
    aa = model(a)
    # print(aa)
