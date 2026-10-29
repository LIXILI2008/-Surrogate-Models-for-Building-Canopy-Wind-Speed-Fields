"""Geometry-to-field surrogates. U-FNO is a 2-D task adaptation, not a paper reproduction."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import features, CHANNELS


def decode(raw, feat):
    # All methods have identical physical heads and inlet initialization.
    d=F.softplus(raw[...,0]+math.log(math.expm1(1.)))-1
    d=torch.where(feat[...,10]>.5,torch.full_like(d,-1),d)
    uvw=raw[...,1:4]+torch.stack((feat[...,11],torch.zeros_like(d),torch.zeros_like(d)),-1)
    k=.01*F.softplus(raw[...,5]+math.log(math.expm1(1.)))+1e-8
    eps=.001*F.softplus(raw[...,6]+math.log(math.expm1(1.)))+1e-9
    return torch.cat((d[...,None],uvw,raw[...,4:5],k[...,None],eps[...,None]),-1)


class ConvBlock(nn.Module):
    def __init__(self,inc,out):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(inc,out,3,padding=1),nn.GroupNorm(4,out),nn.GELU(),
                               nn.Conv2d(out,out,3,padding=1),nn.GroupNorm(4,out),nn.GELU())
    def forward(self,x):return self.net(x)


class UNet(nn.Module):
    def __init__(self,inc,out,width=16):
        super().__init__();w=width
        self.a=ConvBlock(inc,w);self.b=ConvBlock(w,2*w);self.c=ConvBlock(2*w,4*w)
        self.d=ConvBlock(4*w,8*w);self.u2=ConvBlock(12*w,4*w)
        self.u1=ConvBlock(6*w,2*w);self.u0=ConvBlock(3*w,w);self.head=nn.Conv2d(w,out,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
        c=self.u2(torch.cat((F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c),1))
        b=self.u1(torch.cat((F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False),b),1))
        return self.head(self.u0(torch.cat((F.interpolate(b,size=a.shape[-2:],mode='bilinear',align_corners=False),a),1)))


class SpectralConv(nn.Module):
    def __init__(self,width,modes=8):
        super().__init__();self.modes=modes
        # Real/imaginary pairs make real parameter counts and checkpoint portability explicit.
        self.weight=nn.Parameter(torch.randn(2,width,width,modes,modes,2)/(width*2)**.5)
    def forward(self,x):
        b,c,h,w=x.shape;m=min(self.modes,h//2,w//2+1)
        fx=torch.fft.rfft2(x.float(),norm='ortho')
        out=torch.zeros((b,c,h,w//2+1),dtype=fx.dtype,device=x.device)
        weight=torch.view_as_complex(self.weight.contiguous())
        out[:,:,:m,:m]=torch.einsum('bixy,ioxy->boxy',fx[:,:,:m,:m],weight[0,:,:,:m,:m])
        out[:,:,-m:,:m]=torch.einsum('bixy,ioxy->boxy',fx[:,:,-m:,:m],weight[1,:,:,:m,:m])
        return torch.fft.irfft2(out,s=(h,w),norm='ortho')


class FNO(nn.Module):
    def __init__(self,inc,out,width=20,modes=8,ufno=False):
        super().__init__();self.lift=nn.Conv2d(inc,width,1)
        self.spectral=nn.ModuleList([SpectralConv(width,modes) for _ in range(4)])
        self.point=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(4)])
        self.local_u=nn.ModuleList([UNet(width,width,8) for _ in range(2)]) if ufno else None
        self.head=nn.Conv2d(width,out,1)
    def forward(self,x):
        h,w=x.shape[-2:];x=self.lift(x)
        # Nonperiodic physical crop: explicit padding avoids identifying opposite edges.
        x=F.pad(x,(0,8,0,8))
        for i,(s,p) in enumerate(zip(self.spectral,self.point)):
            y=s(x)+p(x)
            if self.local_u is not None and i>=2:y=y+self.local_u[i-2](x)
            x=F.gelu(y)
        return self.head(x[...,:h,:w])


class GridModel(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.backbone=UNet(len(CHANNELS),7) if kind=='unet' else FNO(len(CHANNELS),7,ufno=kind=='ufno')
        nn.init.zeros_(self.backbone.head.weight);nn.init.zeros_(self.backbone.head.bias)
        self.local=nn.Sequential(nn.Conv2d(len(CHANNELS),32,1),nn.SiLU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
    def forward(self,f):
        raw=self.backbone(f);local=self.local(f)*f[:,14:15]
        raw=torch.cat((raw[:,:1]+local,raw[:,1:]),1)
        return decode(raw.permute(0,2,3,1),f.permute(0,2,3,1)).permute(0,3,1,2)


class CoordinateModel(nn.Module):
    def __init__(self,width=128,depth=5):
        super().__init__()
        layers=[nn.Linear(len(CHANNELS),width),nn.Tanh()]
        for _ in range(depth-1):layers += [nn.Linear(width,width),nn.Tanh()]
        self.trunk=nn.Sequential(*layers);self.head=nn.Linear(width,7)
        self.local=nn.Sequential(nn.Linear(len(CHANNELS),64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)
    def from_features(self,f):
        raw=self.head(self.trunk(f));d=raw[...,:1]+self.local(f)*f[...,14:15]
        return decode(torch.cat((d,raw[...,1:]),-1),f)
    def forward(self,q,dims,span):return self.from_features(features(q,dims,span))


def make(kind):
    if kind in ('mlp_data','pinn'):return CoordinateModel()
    if kind in ('unet','fno','ufno'):return GridModel(kind)
    raise ValueError(kind)
