"""Predict four unseen W-L-H combinations with the existing V2.3 and V2.4 models.

The dimensions are deliberately chosen outside the dataset's exact geometry list,
while remaining within the documented per-axis bounds. No reference field exists
for these cases, so this script exports predictions only and never computes error.
"""
from pathlib import Path
import json, sys, hashlib
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
WORKSPACE=ROOT.parent
OUT=ROOT/'deliverables/unseen_geometry_predictions'
V23ROOT=WORKSPACE/'project_v2_3'
V23CK=V23ROOT/'artifacts/height_transfer_numeric_verified/production/best.pt'
V24CK=ROOT/'artifacts/comparison/ufno/bldg_W24_L8_H100/best.pt'

CASES=[
    dict(tag='low_A',W=18.,L=28.,H=22.,z=26.4),
    dict(tag='low_B',W=14.,L=36.,H=35.,z=42.0),
    dict(tag='high_A',W=18.,L=16.,H=60.,z=72.0),
    dict(tag='high_B',W=26.,L=24.,H=90.,z=108.0),
]

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def inlet(z): return 3.*(z/10.)**.4

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # V2.3 is a geometry-conditioned production checkpoint trained on the full
    # verified set. V2.4 is explicitly the available U-FNO comparison fold.
    sys.path.insert(0,str(V23ROOT))
    from fieldgen.train import load as load23
    from fieldgen.predict import infer as infer23
    from fieldgen.geometry import query_grid
    from bench.models import make
    from bench.geometry import grid, features, masks_numpy
    from bench.common import digest as bench_digest

    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    v23,ck23=load23(V23CK,device)
    ck24=torch.load(V24CK,map_location='cpu',weights_only=True)
    v24=make(ck24['architecture']).to(device)
    v24.load_state_dict(ck24['state']);v24.eval()
    records=[]
    for c in CASES:
        dims=np.array([c['W'],c['L'],c['H']],np.float32);z=float(c['z']);size=128
        xy,span=query_grid(dims,size=size)
        # V2.3 and V2.4 use the same physical grid for direct visual comparison.
        bounds=[-span[0]/2+span[0]/(2*size),-span[1]/2+span[1]/(2*size),
                span[0]/2-span[0]/(2*size),span[1]/2-span[1]/(2*size)]
        q=grid(bounds,dims,z,size)
        with torch.no_grad():
            s23=infer23(v23,xy,dims,z,device)
            qt=torch.tensor(q,device=device)
            dt=torch.tensor(dims,device=device).expand(size,size,3)
            st=torch.tensor(span,device=device,dtype=torch.float32).expand(size,size,2)
            f=features(qt,dt,st)
            out=v24(f.permute(2,0,1)[None])[0].permute(1,2,0).cpu().numpy()
        s24=(out[...,0]+1.)*inlet(z)
        projection,roof,fluid=masks_numpy(q,dims)
        s23=np.asarray(s23).copy();s24=np.asarray(s24).copy()
        s23[roof]=0.;s24[roof]=0.
        bg=inlet(z); d23=s23/bg-1.; d24=s24/bg-1.
        vmax=6. if c['H']<=50 else 13.
        # A compact 2x2 plate: two versions in the first row, their relative
        # disturbances in the second. There is intentionally no fake reference.
        fig,ax=plt.subplots(2,2,figsize=(10.5,8.2),layout='constrained')
        plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
        fields=[[s23,s24],[d23,d24]]
        titles=['V2.3 geometry-conditioned MLP','V2.4 U-FNO comparison model']
        labels=['Speed (m/s)','D = U/Uin - 1']
        ims=[]
        for r in range(2):
            for col in range(2):
                v=fields[r][col]
                if r==0:im=ax[r,col].imshow(v,cmap='turbo',vmin=0,vmax=vmax,extent=[bounds[0],bounds[2],bounds[3],bounds[1]],interpolation='nearest',aspect='equal')
                else:im=ax[r,col].imshow(v,cmap='coolwarm',vmin=-.5,vmax=.5,extent=[bounds[0],bounds[2],bounds[3],bounds[1]],interpolation='nearest',aspect='equal')
                ims.append(im)
                ax[r,col].add_patch(Rectangle((-c['L']/2,-c['W']/2),c['L'],c['W'],fill=False,edgecolor='black',linestyle='--',linewidth=.9))
                ax[r,col].set_title(titles[col],fontsize=12,pad=5)
                ax[r,col].set_xlabel('x (m)',fontsize=10);ax[r,col].set_ylabel('y (m)',fontsize=10)
                ax[r,col].tick_params(labelsize=9)
        fig.colorbar(ims[0],ax=ax[0,:].tolist(),fraction=.035,pad=.025,label='m/s',extend='both')
        fig.colorbar(ims[2],ax=ax[1,:].tolist(),fraction=.035,pad=.025,label='dimensionless',extend='both')
        fig.suptitle(f'Unseen geometry prediction | W={c["W"]:g}, L={c["L"]:g}, H={c["H"]:g} m | z={z:g} m | z/H={z/c["H"]:.2f}\nUin={bg:.3f} m/s; same 128×128 physical grid; wind along +x',fontsize=14)
        fig.supxlabel('Dashed rectangle: building projection; z>H projection interior is valid fluid. No reference field is available for this new geometry.',fontsize=10)
        png=OUT/f'{c["tag"]}_W{c["W"]:g}_L{c["L"]:g}_H{c["H"]:g}_z{z:g}.png'
        fig.savefig(png,dpi=180);plt.close(fig)
        npz=png.with_suffix('.npz')
        np.savez_compressed(npz,speed_v23_mps=s23,speed_v24_ufno_mps=s24,
            disturbance_v23=d23,disturbance_v24_ufno=d24,x_m=xy[...,0],y_m=xy[...,1],
            projection=projection,roof_boundary=roof,valid_fluid=fluid,inlet_mps=bg,dims=dims,z=z)
        meta=dict(tag=c['tag'],dims_m=[float(x) for x in dims],z_m=z,z_over_H=z/c['H'],
            inlet_mps=bg,grid=[size,size],field_of_view_m=[float(x) for x in span],
            dimension_is_absent_from_dataset=True,within_per_axis_bounds=True,
            v23=dict(model='geometry-conditioned coordinate MLP',checkpoint=str(V23CK),sha256=sha256(V23CK),production_model=True),
            v24=dict(model='U-FNO',checkpoint=str(V24CK),sha256=sha256(V24CK),production_model=False,held_out=ck24.get('held_out'),
                     note='Available comparison fold used as V2.4 candidate; V2.4 has no selected production checkpoint.'),
            label_status='No reference image or CFD field exists for this unseen W-L-H combination; speed and disturbance are predictions only.',
            roof_policy='Exact-roof policy not used because z>H; building projection is retained as geometry and its interior is valid fluid.',
            color_range_mps=[0.,vmax],disturbance_color_range=[-.5,.5])
        (png.with_suffix('.json')).write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        records.append(meta)
        print(png,flush=True)
    (OUT/'manifest.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE '+str(len(records))+' unseen geometry predictions')

if __name__=='__main__':main()
