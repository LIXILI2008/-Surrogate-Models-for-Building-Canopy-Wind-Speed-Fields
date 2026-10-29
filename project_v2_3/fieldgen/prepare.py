"""Create V2 point supervision from the read-only V1 calibrated/registered cache."""
import argparse
from pathlib import Path
import numpy as np
from .common import ROOT,read,write,digest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default='configs/project.json');a=ap.parse_args()
    path=Path(a.config).resolve();cfg=read(path);src=(path.parent/cfg['source_cache']).resolve();out=(path.parent/cfg['output']).resolve()
    if (out/'manifest.json').exists():raise FileExistsError('V2 dataset exists; use its manifest or choose a new output')
    out.mkdir(parents=True,exist_ok=True);old=read(src/'manifest.json')
    assert old['label']['mode']=='speed' and old['label']['speed_scale']==13
    names=sorted(old['buildings']);dims=np.array([old['buildings'][n]['dims'] for n in names],np.float32)
    val=set(cfg['validation_buildings']);test=set(cfg['test_buildings']);assert not val&test and val|test<=set(names)
    train=set(names)-val-test
    bounds=dict(min=dims.min(0).tolist(),max=dims.max(0).tolist())
    td=np.array([old['buildings'][n]['dims'] for n in train])
    assert np.all(td.min(0)==dims.min(0)) and np.all(td.max(0)==dims.max(0)), 'Keep extrema in training for in-range validation'
    records=[r for r in old['records'] if 1-1e-6<=r['z']/old['buildings'][r['building']]['dims'][2]<=1.5+1e-6]
    n=cfg['points_per_slice'];pool=np.lib.format.open_memmap(out/'points.npy',mode='w+',dtype='float32',shape=(len(records),n,3))
    rng=np.random.default_rng(cfg['seed']);geometry={};metadata=[]
    for name in names:
        with np.load(src/'cache'/name/'geometry.npz') as g:fp=g['footprint'].copy()
        y,x=np.where(fp);width=x.max()-x.min()+1;height=y.max()-y.min()+1
        geometry[name]=dict(cx=float((x.min()+x.max())/2),cy=float((y.min()+y.max())/2),
            dx=float(old['buildings'][name]['dims'][1]/width),dy=float(old['buildings'][name]['dims'][0]/height))
    for i,r in enumerate(records):
        name=r['building'];w,l,h=old['buildings'][name]['dims'];g=geometry[name]
        with np.load(src/r['cache']) as cache:y=cache['target'][...,0].astype(np.float32);valid=cache['valid'].copy()
        yy,xx=np.indices(valid.shape);xm=(xx-g['cx'])*g['dx'];ym=(yy-g['cy'])*g['dy']
        # The roof is a separately prescribed stationary no-slip boundary, not fluid truth.
        if abs(r['z']-h)<1e-6:valid&=~((abs(xm)<=l/2)&(abs(ym)<=w/2))
        rr,cc=np.where(valid & np.isfinite(y));assert len(rr),r['source']
        idx=rng.integers(len(rr),size=n);sr,sc=rr[idx],cc[idx]
        pool[i,:,0]=xm[sr,sc];pool[i,:,1]=ym[sr,sc];pool[i,:,2]=y[sr,sc]
        metadata.append(dict(building=name,building_index=names.index(name),dims=[w,l,h],z=r['z'],ratio=r['z']/h,
            split='validation' if name in val else 'test' if name in test else 'train',source=r['source'],source_sha256=r['source_sha256'],
            cache=r['cache'],cache_sha256=digest(src/r['cache']),valid_pixels=int(len(rr)),
            observed_xy_bounds=[float(xm[rr,cc].min()),float(ym[rr,cc].min()),float(xm[rr,cc].max()),float(ym[rr,cc].max())]))
        if (i+1)%200==0:print(f'Prepared {i+1}/{len(records)} slices',flush=True)
    pool.flush();del pool
    palettes=[]
    for b in old['config']['calibration']['bands']:
        p=ROOT/'configs'/b['lut_csv'];palettes.append(dict(filename=p.name,sha256=digest(p)))
    write(out/'manifest.json',dict(schema=2,config=cfg,source_cache=str(src),source_manifest_sha256=digest(src/'manifest.json'),
        points_sha256=digest(out/'points.npy'),names=names,dims=dims.tolist(),bounds=bounds,geometry=geometry,
        splits=dict(train=sorted(train),validation=sorted(val),test=sorted(test)),records=metadata,palettes=palettes,
        normalization='Existing normalized physical cache cast to float32; no second division.',
        label_limitations='Inherited V1 image-derived speed and estimated above-roof registration; finite coverage and uncertain alignment remain.',
        roof_policy='Exact H footprint is a prescribed no-slip boundary; excluded from scored fluid pixels.'))
    print(f'Prepared {len(records)} slices, buildings {len(train)}/{len(val)}/{len(test)}',flush=True)


if __name__=='__main__':main()
