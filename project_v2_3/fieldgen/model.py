import torch
from torch import nn
from .geometry import features


class Field(nn.Module):
    """Dimension-conditioned shared field; no geometry lookup/embedding table."""
    def __init__(self,width=96,depth=4,output_mode='legacy',factorized_rank=0):
        super().__init__()
        if output_mode not in ('legacy','reference'):raise ValueError(output_mode)
        self.output_mode=output_mode
        if factorized_rank<0:raise ValueError('factorized_rank must be nonnegative')
        self.factorized_rank=factorized_rank
        self.input=nn.Linear(25,width)
        self.layers=nn.ModuleList([nn.Linear(width,width) for _ in range(depth-1)])
        self.condition=nn.Sequential(nn.Linear(5,64),nn.SiLU(),nn.Linear(64,factorized_rank or 2*width))
        self.head=nn.Linear(width,factorized_rank or 1)
        self.local=nn.Sequential(nn.Linear(30,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
        if factorized_rank:
            # Zero coefficients preserve inlet initialization; nonzero bases
            # provide the branch with a learning signal from the first step.
            nn.init.zeros_(self.condition[-1].weight);nn.init.zeros_(self.condition[-1].bias)
        else:
            nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.local[-1].weight);nn.init.zeros_(self.local[-1].bias)

    def forward(self,xy,dims,ratio):
        f,c,bg,gate,roof,weight=features(xy,dims,ratio)
        if self.factorized_rank:
            t=torch.nn.functional.silu(self.input(f))
        else:
            scale,shift=self.condition(c).chunk(2,-1)
            t=torch.nn.functional.silu(self.input(f)*(1+.2*scale.tanh())+.2*shift)
        for layer in self.layers:t=torch.nn.functional.silu(layer(t)+.1*t)
        if self.factorized_rank:
            global_factor=(self.head(t)*self.condition(c)).sum(-1)/self.factorized_rank**.5
        else:
            global_factor=self.head(t)[:,0]
        log_factor=global_factor+gate*self.local(torch.cat((f,c),-1))[:,0]
        # Equivalent residual form: speed = inlet + inlet*expm1(log_factor).
        # Positive total magnitude with signed disturbance; exact inlet at initialization.
        if self.output_mode=='reference':
            # Learn disturbance relative to U(H). At initialization pred=U(z).
            # Softplus keeps speed positive without the dead gradient of clipping.
            ref=3*(dims[:,2]/10).pow(.4)/13
            base=bg/ref
            inverse=base+torch.log(-torch.expm1(-5*base))/5
            pred=ref*torch.nn.functional.softplus(inverse+log_factor,beta=5)
        else:
            pred=bg*torch.exp(log_factor.clamp(-10,2))
        pred=torch.where(roof,torch.zeros_like(pred),pred)
        return pred,weight


class Ensemble(nn.Module):
    def __init__(self,members):
        super().__init__();self.members=nn.ModuleList(members)

    def forward(self,xy,dims,ratio):
        result=[m(xy,dims,ratio) for m in self.members]
        return torch.stack([p for p,_ in result]).mean(0),result[0][1]
