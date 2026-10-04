import math
import torch, torch.nn as nn, torch.nn.functional as F

def cbr(i, o, k=3, s=1, g=1):
    return nn.Sequential(nn.Conv2d(i, o, k, s, k // 2, groups=g, bias=False),
                         nn.BatchNorm2d(o), nn.SiLU(inplace=True))

class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = cbr(c, c, 3)
        self.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c))
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        return self.act(x + self.b(self.a(x)))

class Net(nn.Module):
    """Multi-task: CenterNet-ish detection head + occupancy grid + count/zone/area heads."""
    def __init__(self, wm=1.0, ncls=13):
        super().__init__()
        c = [int(x * wm) for x in (24, 48, 80, 128, 192)]
        self.s2 = nn.Sequential(cbr(3, c[0], 3, 2), Res(c[0]))                 # /2
        self.s4 = nn.Sequential(cbr(c[0], c[1], 3, 2), Res(c[1]))              # /4
        self.s8 = nn.Sequential(cbr(c[1], c[2], 3, 2), Res(c[2]), Res(c[2]))   # /8
        self.s16 = nn.Sequential(cbr(c[2], c[3], 3, 2), Res(c[3]), Res(c[3]))  # /16
        self.s32 = nn.Sequential(cbr(c[3], c[4], 3, 2), Res(c[4]))             # /32
        self.l16 = cbr(c[3], c[2], 1)
        self.l32 = cbr(c[4], c[2], 1)
        self.fuse = nn.Sequential(cbr(c[2], c[2], 3), Res(c[2]))               # at /8
        self.hm = nn.Sequential(cbr(c[2], c[2], 3), nn.Conv2d(c[2], 1, 1))
        self.wh = nn.Sequential(cbr(c[2], c[2], 3), nn.Conv2d(c[2], 2, 1))
        self.off = nn.Conv2d(c[2], 2, 1)
        self.grid = nn.Sequential(cbr(c[3], c[3], 3), nn.Conv2d(c[3], 1, 1))   # -> pooled to 10x12
        self.gp = nn.Sequential(nn.Linear(c[4] + c[3], 256), nn.SiLU(inplace=True), nn.Dropout(0.15))
        self.cnt = nn.Linear(256, ncls)
        self.zon = nn.Linear(256, 3)
        self.abn = nn.Linear(256, 4)
        self.hm[-1].bias.data.fill_(-3.5)
        self.grid[-1].bias.data.fill_(-1.5)

    def forward(self, x):
        x2 = self.s2(x); x4 = self.s4(x2); x8 = self.s8(x4); x16 = self.s16(x8); x32 = self.s32(x16)
        f = x8 + F.interpolate(self.l16(x16), size=x8.shape[-2:], mode="nearest") \
               + F.interpolate(self.l32(x32), size=x8.shape[-2:], mode="nearest")
        f = self.fuse(f)
        g = F.adaptive_avg_pool2d(self.grid(x16), (10, 12)).squeeze(1)
        pooled = torch.cat([F.adaptive_avg_pool2d(x32, 1).flatten(1),
                            F.adaptive_avg_pool2d(x16, 1).flatten(1)], 1)
        h = self.gp(pooled)
        return {"hm": self.hm(f), "wh": self.wh(f), "off": self.off(f), "grid": g,
                "cnt": self.cnt(h), "zon": self.zon(h), "abn": self.abn(h)}


def focal(pred, gt, a=2.0, b=4.0):
    """CornerNet/CenterNet penalty-reduced focal loss. pred=logits, gt in [0,1]."""
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float(); neg = 1 - pos
    nw = torch.pow(1 - gt, b)
    lp = torch.log(p) * torch.pow(1 - p, a) * pos
    ln = torch.log(1 - p) * torch.pow(p, a) * nw * neg
    npos = pos.sum()
    return -(lp.sum() + ln.sum()) / npos.clamp(min=1)
