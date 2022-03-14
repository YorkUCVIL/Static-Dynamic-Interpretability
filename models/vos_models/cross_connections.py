import torch.nn.functional as F
import torch.nn as nn
import torch

class CoAttention(nn.Module):
    def __init__(self, channel, normalization_fn='tanh'):
        super(CoAttention, self).__init__()
        print("====> Creating CoAttention Cross Connections")
        self.normalization_fn = normalization_fn

        d = channel // 16
        self.proja = nn.Conv2d(channel, d, kernel_size=1)
        self.projb = nn.Conv2d(channel, d, kernel_size=1)

        self.bottolneck1 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.bottolneck2 = nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.proj1 = nn.Conv2d(channel, 1, kernel_size=1)
        self.proj2 = nn.Conv2d(channel, 1, kernel_size=1)

        self.bna = nn.BatchNorm2d(channel)
        self.bnb = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        _, Zb = self.forward_co(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa)
        Pb = F.relu(Qb_1 + Qb)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_2 + Pb)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_3, Qb_3)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_3 + Pb)

        # cascade 4
        Qa_4, Qb_4 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_4, Qb_4)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Qb_4 + Pb)

        # cascade 5
        Qa_5, Qb_5 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_5, Qb_5)

        return Za, Zb, Qa_5, Qb_5

    def forward_sa(self, Qa, Qb):
        Aa = self.proj1(Qa)
        Ab = self.proj2(Qb)

        n, c, h, w = Aa.shape
        Aa = Aa.view(-1, h*w)
        Ab = Ab.view(-1, h*w)

        Aa = F.softmax(Aa)
        Ab = F.softmax(Ab)

        Aa = Aa.view(n, c, h, w)
        Ab = Ab.view(n, c, h, w)

        Qa_attened = Aa * Qa
        Qb_attened = Ab * Qb

        return Qa_attened, Qb_attened

    def forward_co(self, Qa, Qb):
        Qa_low = self.proja(Qa)
        Qb_low = self.projb(Qb)

        N, C, H, W = Qa_low.shape
        Qa_low = Qa_low.view(N, C, H * W)
        Qb_low = Qb_low.view(N, C, H * W)
        Qb_low = torch.transpose(Qb_low, 1, 2)

        L = torch.bmm(Qb_low, Qa_low)

        if self.normalization_fn == 'tanh':
            Aa = F.tanh(L)
            Ab = torch.transpose(Aa, 1, 2)
        elif self.normalization_fn == 'softmax':
            Aa = F.softmax(L, dim=2)

            Ab = F.softmax(L, dim=1)
            Ab = torch.transpose(Ab, 1, 2)

        N, C, H, W = Qa.shape

        Qa_ = Qa.view(N, C, H * W)
        Qb_ = Qb.view(N, C, H * W)

        Za = torch.bmm(Qb_, Aa)
        Zb = torch.bmm(Qa_, Ab)
        Za = Za.view(N, C, H, W)
        Zb = Zb.view(N, C, H, W)

        Za = F.normalize(Za)
        Zb = F.normalize(Zb)

        return Za, Zb

class RGating(nn.Module):
    def __init__(self, in_planes):
        super(RGating, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_planes*2, 2, kernel_size=1), nn.Sigmoid())

    def forward(self, img, flow):
        gate = self.conv(torch.cat([img, flow], 1))
        return img.mul(gate[:,:1,:,:])+img, flow.mul(gate[:,1:,:,:])+flow

class CoAttentionGated(CoAttention):
    def __init__(self, channel, normalization_fn='tanh'):
        super(CoAttentionGated, self).__init__(channel, normalization_fn)
        self.rgating = RGating(channel)
        print("====> Creating CoAttention Gated Cross Connections")

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        _, Zb = self.forward_co(Qa_1, Qb_1)
        Za_1, _ = self.rgating(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa + Za_1)
        Pb = F.relu(Qb_1 + Qb)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_2, Qb_2)
        Za_1, _ = self.rgating(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Qb_2 + Pb)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_3, Qb_3)
        Za_1, _ = self.rgating(Qa_3, Qb_3)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Qb_3 + Pb)

        # cascade 4
        Qa_4, Qb_4 = self.forward_sa(Pa, Pb)
        _, Zb = self.forward_co(Qa_4, Qb_4)
        Za_1, _ = self.rgating(Qa_4, Qb_4)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Qb_4 + Pb)

        # cascade 5
        Qa_5, Qb_5 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_5, Qb_5)
        Za_1, Zb_1 = self.rgating(Qa_5, Qb_5)

        Zb = Zb + Za_1
        Za = Za + Zb_1

        return Za, Zb, Qa_5, Qb_5

class CoAttentionRecip(CoAttention):
    def __init__(self, channel, normalization_fn='tanh'):
        super(CoAttentionRecip, self).__init__(channel, normalization_fn)
        print("====> Creating CoAttention Reciprocal Cross Connections")

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        Za, Zb = self.forward_co(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa)
        Pb = F.relu(Za + Qb)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Za + Pb)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_3, Qb_3)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Za + Pb)

        # cascade 4
        Qa_4, Qb_4 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_4, Qb_4)

        Pa = F.relu(Zb + Pa)
        Pb = F.relu(Za + Pb)

        # cascade 5
        Qa_5, Qb_5 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_5, Qb_5)

        return Za, Zb, Qa_5, Qb_5

class CoAttentionGatedRecip(CoAttention):
    def __init__(self, channel, normalization_fn='tanh'):
        super(CoAttentionGatedRecip, self).__init__(channel, normalization_fn='tanh')
        self.rgating = RGating(channel)
        print("====> Creating CoAttention Reciprocal Gated Cross Connections")

    def forward(self, Qa, Qb):
        # cascade 1
        Qa_1, Qb_1 = self.forward_sa(Qa, Qb)
        Za, Zb = self.forward_co(Qa_1, Qb_1)
        Za_1, Zb_1 = self.rgating(Qa_1, Qb_1)

        Pa = F.relu(Zb + Qa + Za_1)
        Pb = F.relu(Za + Qb + Zb_1)

        # cascade 2
        Qa_2, Qb_2 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_2, Qb_2)
        Za_1, Zb_1 = self.rgating(Qa_2, Qb_2)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Za + Pb + Zb_1)

        # cascade 3
        Qa_3, Qb_3 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_3, Qb_3)
        Za_1, Zb_1 = self.rgating(Qa_3, Qb_3)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Za + Pb + Zb_1)

        # cascade 4
        Qa_4, Qb_4 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_4, Qb_4)
        Za_1, Zb_1 = self.rgating(Qa_4, Qb_4)

        Pa = F.relu(Zb + Pa + Za_1)
        Pb = F.relu(Za + Pb + Zb_1)

        # cascade 5
        Qa_5, Qb_5 = self.forward_sa(Pa, Pb)
        Za, Zb = self.forward_co(Qa_5, Qb_5)
        Za_1, Zb_1 = self.rgating(Qa_5, Qb_5)

        Za = Za + Zb_1
        Zb = Zb + Za_1

        return Za, Zb, Qa_5, Qb_5
