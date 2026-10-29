"""Unified entry point for the six original methods and geometry adaptation."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from study import ART, METHODS, SEEDS, read, write, digest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='command',required=True)
    sub.add_parser('status')
    baseline=sub.add_parser('baseline')
    baseline.add_argument('action',choices=['prepare','run','report','finalize'])
    pr=sub.add_parser('predict')
    pr.add_argument('--method',choices=METHODS,default='v23_mlp')
    pr.add_argument('--dims',type=float,nargs=3,required=True,metavar=('W','L','H'))
    pr.add_argument('--z',type=float,required=True)
    pr.add_argument('--seed',type=int,choices=SEEDS,default=SEEDS[0])
    pr.add_argument('--output',type=Path,required=True)
    ad=sub.add_parser('adapt');ad.add_argument('action',choices=['prepare','run','report','status'])
    args=ap.parse_args()
    if args.command=='status':
        print(read(ART/'progress.json'))
        path=ART/'adaptation_v1/status.json'
        if path.exists():print(read(path))
    elif args.command=='baseline':
        import subprocess,sys
        subprocess.run([sys.executable,str(Path(__file__).with_name('study.py')),args.action],check=True)
    elif args.command=='adapt':
        from adaptation import dispatch
        dispatch(args.action)
    else:
        from analysis_pipeline import grid,load_model,infer
        w,l,h=args.dims
        if min(w,l,h)<=0 or not h<args.z<1.5*h:
            raise ValueError('Positive W,L,H and strict H < z < 1.5H are required')
        p=read(ART/'protocol.json')
        job=dict(method=args.method,seed=args.seed,train=p['training_buildings'],test=[],
                 key=f'production/{args.method}/seed_{args.seed}')
        model,ck,device=load_model(job);xy,bounds=grid(args.dims)
        speed=infer(model,args.method,xy,args.dims,args.z,bounds,device)
        out=args.output.resolve()
        if out.exists():raise FileExistsError(out)
        out.mkdir(parents=True)
        np.savez_compressed(out/'field.npz',xy_m=xy,speed_mps=speed,dims_WLH_m=args.dims,z_m=args.z)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
        im=ax.pcolormesh(xy[...,0],xy[...,1],speed,shading='auto',cmap='turbo')
        ax.add_patch(Rectangle((-l/2,-w/2),l,w,fill=False,linestyle='--',color='black'))
        ax.set(aspect='equal',xlabel='x (m)',ylabel='y (m)',title=f'{args.method}: W={w:g}, L={l:g}, H={h:g}; z={args.z:g} m')
        fig.colorbar(im,ax=ax,label='Speed (m/s)');fig.savefig(out/'field.png',dpi=180);plt.close(fig)
        write(out/'provenance.json',dict(method=args.method,seed=args.seed,dims_WLH_m=args.dims,z_m=args.z,
            checkpoint_sha256=digest(ART/job['key']/'best.pt'),grid_size=128,bounds=bounds,
            units='m/s',inflow='U(z)=3*(z/10)^0.4 m/s; wind +x',
            interpretation='Model prediction only; no CFD label or accuracy claim for this query'))
        print(out/'field.npz')


if __name__=='__main__':main()
