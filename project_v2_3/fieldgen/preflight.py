"""Short training/physical-label inference integration check, separate from scores."""
import argparse
from pathlib import Path
import numpy as np
import torch
from .common import read,write
from .height_transfer import plan_for,fit
from .labels import load_label
from .train import load
from .predict import infer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--output',required=True);a=ap.parse_args()
    ds=Path(a.dataset);out=Path(a.output)
    if (out/'complete.json').exists():return
    meta=read(ds/'manifest.json');plan=plan_for(meta)
    plan.update(source_steps=20,joint_steps=200,adaptation_steps=180,flexible_steps=180,local_steps=5)
    held=plan['high'][0]
    fit(ds,out/'source',plan,'source')
    fit(ds,out/'flexible',plan,'transfer_flexible',held_out=held,source=out/'source/best.pt')
    model,ck=load(out/'flexible/best.pt','cpu');torch.set_num_threads(4)
    rows=[r for r in meta['records'] if r['building']==held]
    checks=[]
    for r in (rows[0],rows[1],rows[-1]):
        xy,y,valid=load_label(meta,r,ds);p=infer(model,xy,r['dims'],r['z'],'cpu')
        assert np.isfinite(p).all() and (p>=0).all() and held not in ck['trained_buildings']
        w,l,h=r['dims'];roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(r['z']-h)<1e-6)
        if roof.any():assert (p[roof]==0).all()
        checks.append(dict(z=r['z'],fluid_pixels=int(valid.sum()),finite_prediction=True))
    write(out/'complete.json',dict(state='PASSED',purpose='Integration smoke test only; do not interpret as benchmark performance',checks=checks))

if __name__=='__main__':main()
