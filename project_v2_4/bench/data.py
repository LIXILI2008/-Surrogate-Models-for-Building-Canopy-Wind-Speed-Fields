import numpy as np
import torch
from .common import read,digest
from .geometry import grid,features


class Dataset:
    def __init__(self,path,verify=True):
        self.path=path;self.meta=read(path/'manifest.json');self.rows=self.meta['records']
        if verify:
            for f,v in self.meta['array_sha256'].items():assert digest(path/f)==v
        self.target=np.load(path/'targets.npy',mmap_mode='r')
        self.valid=np.load(path/'valid.npy',mmap_mode='r')
        self.aux_valid=np.load(path/'aux_valid.npy',mmap_mode='r')
        self.ids={n:[i for i,r in enumerate(self.rows) if r['building']==n] for n in self.meta['names']}
        self.dimensions=dict(zip(self.meta['names'],self.meta['dims']))
    def batch(self,ids,device):
        rr=[self.rows[int(i)] for i in ids]
        q=np.stack([grid(r['bounds'],r['dims'],r['z'],self.meta['size']) for r in rr])
        dims=torch.tensor([r['dims'] for r in rr],device=device,dtype=torch.float32)
        span=torch.tensor([r['span'] for r in rr],device=device,dtype=torch.float32)
        qt=torch.tensor(q,device=device)
        f=features(qt,dims[:,None,None,:].expand_as(qt),span[:,None,None,:].expand(*qt.shape[:-1],2))
        return dict(q=qt,dims=dims,span=span,f=f,target=torch.tensor(self.target[ids].copy().transpose(0,2,3,1),device=device),
                    valid=torch.tensor(self.valid[ids].copy(),device=device),
                    aux_valid=torch.tensor(self.aux_valid[ids].copy(),device=device))
    def sample(self,names,rng):
        out=[]
        for high in (False,True):
            candidates=[n for n in names if (self.dimensions[n][2]>50)==high]
            name=str(rng.choice(candidates));ids=self.ids[name]
            strata=[[i for i in ids if min(2,int((self.rows[i]['ratio']-1)*6))==j] for j in range(3)]
            strata=[s for s in strata if s];chosen=strata[int(rng.integers(len(strata)))]
            out.append(int(rng.choice(chosen)))
        return out


def flat_sample(batch,indices):
    b=len(indices);s=batch['f'].shape[1];n=indices.shape[1]
    result={}
    index=torch.tensor(indices,device=batch['f'].device)
    for key in ('q','f','target'):
        a=batch[key].reshape(b,s*s,-1);result[key]=a.gather(1,index[...,None].expand(-1,-1,a.shape[-1]))
    for key in ('valid','aux_valid'):
        result[key]=batch[key].reshape(b,s*s).gather(1,index)
    result['dims']=batch['dims'][:,None,:].expand(b,n,3)
    result['span']=batch['span'][:,None,:].expand(b,n,2)
    return result
