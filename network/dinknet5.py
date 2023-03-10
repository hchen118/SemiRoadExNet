import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)

class CC_module(nn.Module):
    def __init__(self, in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()  # nChw

        proj_query = self.query_conv(x)  # nchw
        # nchw->nwch->(n*w)ch->(n*w)hc
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width,-1,height).permute(0,2,1)
        # nchw->nhcw->(n*h)*c*w>(n*h)wc
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height,-1,width).permute(0,2,1)

        proj_key = self.key_conv(x)  # nchw
        # nchw->nwch->(n*w)ch
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # nchw->nhcw->(n*h)cw
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        proj_value = self.value_conv(x)  # nChw
        # nChw->nwCh->(n*w)Ch
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        # nChw->nhCw->(n*h)Cw
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)

        # (n*w)hh->nwhh->nhwh
        energy_H=(torch.bmm(proj_query_H,proj_key_H)+self.INF(m_batchsize,height,width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        # (n*h)ww->nhww
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)

        # nhwh->nwhh->(n*w)hh
        att_H = self.softmax(energy_H).permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)
        # nhww->(n*h)ww
        att_W = self.softmax(energy_W).contiguous().view(m_batchsize * height, width, width)
        # (n*w)Ch->nwCh->nChw
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        # (n*h)Cw->nhCw->nChw
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)
        # print(out_H.size(),out_W.size())
        return self.gamma * (out_H + out_W) + x

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # nchw->nc11
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  # nc11->nc1->n1c->nc11
        y = self.sigmoid(y)

        return  x*y.expand_as(x)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity


    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, pool_old=1):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        if pool_old:
            self.firstmaxpool = resnet.maxpool
        else:
            self.firstmaxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=True)
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3 + pool_old, 2, pool_old)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 4 - pool_old, padding=pool_old)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=pool_old)

    def forward(self, x):  # 4*3*512*512
        # Encoder
        x = self.firstconv(x)  # 4*64*256*256
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)  # 4*64*128*128
        x = self.eca0(x)
        x=self.cc0(x)

        e1 = self.encoder1(x)  # 4*64*128*128
        e1 = self.eca1(e1)
        e1=self.cc1(e1)

        e2 = self.encoder2(e1)  # 4*128*64*64
        e2=self.eca2(e2)
        e2 = self.cc2(e2)

        e3 = self.encoder3(e2)  # 4*256*32*32

        e4 = self.encoder4(e3)  # 4*512*16*16

        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3  # 4*256*32*32
        d3 = self.decoder3(d4) + e2  # 4*128*64*64
        d2 = self.decoder2(d3) + e1  # 4*64*128*128
        d1 = self.decoder1(d2)  # 4*64*256*256

        out = self.finaldeconv1(d1)  # 4*32*512*512
        out = self.finalrelu1(out)

        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        # out=nn.Dropout2d(0.5)(out)
        out = self.finalconv3(out)  # 4*1*512*512

        return torch.sigmoid(out)





