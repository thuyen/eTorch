import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
            #nn.Dropout(0.2))


class Model(nn.Module):
    def __init__(self):
        n, m = 16, 1
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.convd1 = conv3x3(1*m, 1*n)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd3 = conv3x3(2*n, 4*n)

        self.convu2 = conv3x3(6*n, 2*n)
        self.convu1 = conv3x3(3*n, 1*n)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x1 = x
        x1 = self.convd1(x1)

        x2 = self.maxpool(x1)
        x2 = self.convd2(x2)

        x3 = self.maxpool(x2)
        x3 = self.convd3(x3)

        y2 = self.upsample(x3)
        y2 = torch.cat([x2, y2], 1)
        y2 = self.convu2(y2)

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        y1 = self.convu1(y1)

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)

        return y1

