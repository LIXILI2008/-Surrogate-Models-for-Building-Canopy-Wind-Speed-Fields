"""Render one case with physical rulers to audit geometry and pixel aspect."""
from __future__ import annotations
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from study import ART, METHODS, SEEDS, read
from analysis_pipeline import Labels
from reporting import LABELS


def run():
    p=read(ART/'protocol.json');name=next(n for n in p['external_buildings'] if n=='bldg_W34_L12_H66')
    ds=Labels('production',[name]);ids=[i for i,r in enumerate(ds.rows) if r['building']==name]
    i=min(ids,key=lambda i:abs(ds.rows[i]['z']/ds.rows[i]['dims'][2]-1.05));row=ds.rows[i]
    xy,bounds,truth,valid,_=ds.get(row);w,l,h=row['dims']
    methods=[m for m in METHODS if (ART/f'production/{m}/seed_{SEEDS[0]}/predictions.npy').exists()]
    panels=[('CFD reference',truth,'truth')]
    for m in methods:panels.append((LABELS[m],np.load(ART/f'production/{m}/seed_{SEEDS[0]}/predictions.npy',mmap_mode='r')[i],'prediction'))
    n=len(panels);fig,axes=plt.subplots(1,n,figsize=(4.1*n,9),sharex=True,sharey=True,layout='constrained')
    ex=[float(bounds[0]),float(bounds[2]),float(bounds[1]),float(bounds[3])]
    cmap='turbo';vmax=13
    for ax,(title,field,_) in zip(axes,panels):
        im=ax.pcolormesh(xy[...,0],xy[...,1],np.where(valid,field,np.nan),shading='nearest',
            cmap=cmap,vmin=0,vmax=vmax,rasterized=True)
        ax.add_patch(Rectangle((-l/2,-w/2),l,w,fill=False,edgecolor='white',linestyle='--',linewidth=1.3))
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks(np.linspace(bounds[0],bounds[2],6));ax.set_yticks(np.linspace(bounds[1],bounds[3],7))
        ax.tick_params(labelsize=8);ax.grid(color='white',alpha=.3,linewidth=.35)
        ax.set_xlabel('x streamwise (m)',fontsize=9);ax.set_title(title,fontsize=10)
    axes[0].set_ylabel('y crosswind (m)',fontsize=9)
    fig.colorbar(im,ax=axes.tolist(),shrink=.76,label='wind speed (m/s)')
    dx=(bounds[2]-bounds[0])/127;dy=(bounds[3]-bounds[1])/127
    rx=(bounds[2]-bounds[0])/(bounds[3]-bounds[1]);rbuilding=l/w
    fig.suptitle(f"Coordinate/aspect audit · U01 · Fluent footprint DX=L={l:.2f} m, DY=W={w:.2f} m, H={h:g} m\n"
        f"z={row['z']:g} m ({row['z']/h:.4f}H) · query bounds x=[{bounds[0]:g},{bounds[2]:g}] m, y=[{bounds[1]:g},{bounds[3]:g}] m\n"
        f"Flow +x left→right · domain width/height={rx:.4f}; footprint L/W={rbuilding:.4f}; 128×128 nodes, Δx={dx:.4f} m, Δy={dy:.4f} m")
    out=ART/'interim_completed_models'/'gallery_physical_scale'/'coordinate_scale_check_U01_z69.png'
    fig.savefig(out,dpi=170);plt.close(fig)
    # A single reference map at a readable scale, with an explicit 10 m ruler.
    fig,ax=plt.subplots(figsize=(8,14),layout='constrained')
    im=ax.pcolormesh(xy[...,0],xy[...,1],np.where(valid,truth,np.nan),shading='nearest',
        cmap=cmap,vmin=0,vmax=vmax,rasterized=True)
    ax.add_patch(Rectangle((-l/2,-w/2),l,w,fill=False,edgecolor='white',linestyle='--',linewidth=2))
    ax.set_aspect('equal',adjustable='box');ax.set_xticks(np.arange(-24,37,6));ax.set_yticks(np.arange(-51,52,17))
    ax.tick_params(labelsize=11);ax.grid(color='white',alpha=.25,linewidth=.6)
    ax.set_xlabel('x streamwise (m)',fontsize=12);ax.set_ylabel('y crosswind (m)',fontsize=12)
    ax.set_title(f"CFD reference · U01 · z={row['z']:g} m · H={h:g} m · W={w:.2f} m · L={l:.2f} m\n"
        'Flow +x, from left to right; dashed line is the actual roof footprint',fontsize=13)
    ax.plot([20,30],[-44,-44],color='white',lw=4,solid_capstyle='butt')
    ax.text(25,-42,'10 m',ha='center',va='bottom',color='white',fontsize=11,
        bbox=dict(facecolor='black',alpha=.65,edgecolor='none',pad=2))
    fig.colorbar(im,ax=ax,shrink=.82,label='wind speed (m/s)')
    single=out.with_name('U01_z69_CFD_reference_meter_scale.png');fig.savefig(single,dpi=170);plt.close(fig)
    payload=dict(building=row['building'],label=row['label'],dims_W_L_H_m=[float(w),float(l),float(h)],z_m=float(row['z']),
        domain_bounds_m=list(map(float,bounds)),domain_width_height_ratio=float(rx),footprint_L_W_ratio=float(rbuilding),
        equal_metre_aspect=True,grid=[128,128],dx_m=float(dx),dy_m=float(dy),display_interpolation='nearest')
    out.with_suffix('.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    index=out.parent/'index.html'
    s=index.read_text(encoding='utf-8')
    note=(f'<p class="scale-audit"><strong>坐标比例核对图：</strong><a href="{single.name}">U01 CFD参考场单图</a>；<a href="{out.name}">所有已完成模型的刻度对照</a>。'
          f'含米制刻度、来流方向标注、实际建筑轮廓和像素间距。'
          f'域宽/高={rx:.3f}，建筑L/W={rbuilding:.3f}，每格Δx={dx:.3f} m、Δy={dy:.3f} m。'
          '网格等节点数但非等米间距；视野按x/L=[-2,3]设置，建筑位于从左至右40%处以保留较长下游尾流，这种偏心是视野定义造成的。</p>')
    s=re.sub(r'<p(?: class="scale-audit")?><strong>坐标比例核对图：</strong>.*?</p>','',s)
    index.write_text(s.replace('<h1>',note+'<h1>',1),encoding='utf-8')
    print(json.dumps(payload,indent=2),flush=True)


if __name__=='__main__':run()
