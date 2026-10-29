"""Real GPU short runs + held-out inference; never used as research results."""
import time
from .common import ROOT,write
from .train import fit,physics_loss
from .evaluate import evaluate
from .data import Dataset
from .models import make
import numpy as np
import torch


def main():
    root=ROOT/'artifacts/preflight';root.mkdir(parents=True,exist_ok=True)
    dataset=ROOT/'artifacts/dataset_128';name='bldg_W11_L12_H70';records=[]
    for method in ['unet_si','unet','fno','ufno','mlp_data','pinn']:
        start=time.time();run=root/'sampling_v2'/method
        fit(dataset,run,method,name,steps=8,local_steps=2)
        m=evaluate(dataset,run,name)
        assert np.isfinite(m['speed']['rmse'])
        records.append(dict(method=method,seconds=time.time()-start,rmse_smoke_only=m['speed']['rmse']))
        write(root/'status.json',dict(state='RUNNING',completed=records))
    ds=Dataset(dataset);model=make('pinn').cuda()
    torch.nn.init.normal_(model.head.weight,std=.01)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    for step in range(3):
        loss,values=physics_loss(model,ds,[n for n in ds.meta['names'] if n!=name],np.random.default_rng(step),'cuda')
        assert torch.isfinite(loss);optimizer.zero_grad();loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    write(root/'status.json',dict(state='PASSED',completed=records,pinn_cuda_ad_backward=True,pde_mse_last=values))
    print('PREFLIGHT PASSED',flush=True)


if __name__=='__main__':main()
