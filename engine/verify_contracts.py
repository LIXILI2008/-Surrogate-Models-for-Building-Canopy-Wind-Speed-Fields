"""Targeted invariants for grouped exclusion, grid masks and height aggregation."""
from pathlib import Path
import numpy as np
from study import prepare,ART,EXTERNAL,OLD24,read,family,METHODS
from analysis_pipeline import grid,region_masks,height_weights,scalar,screening,Labels

def main():
    p=prepare();jobs=read(ART/'jobs.json');assert len(jobs)==56 and len(METHODS)==7
    dims=dict(zip(read(OLD24/'manifest.json')['names'],read(OLD24/'manifest.json')['dims']))
    tested=[]
    for f in p['folds']:
        assert not set(f['train']) & set(f['test'])
        assert {family(dims[n]) for n in f['train']}.isdisjoint({family(dims[n]) for n in f['test']})
        tested+=f['test']
    assert sorted(tested)==p['training_buildings']
    for j in jobs:
        assert not set(j['train'])&set(p['external_buildings'])
        if j['kind']=='production':assert j['train']==p['training_buildings']
    ds=Labels('production',p['external_buildings']);assert len(ds.rows)==919
    assert all(1<r['z']/r['dims'][2]<1.5 for r in ds.rows)
    for name in p['external_buildings']:
        row=next(r for r in ds.rows if r['building']==name);xy,bounds,y,v,dist=ds.get(row)
        assert xy.shape==(128,128,2) and y.shape==v.shape==(128,128)
        masks=region_masks(xy,row['dims']);assert masks['roof'].any()
        assert np.all(masks['roof_core']|masks['roof_edge']==masks['roof'])
        assert not (masks['roof_core']&masks['roof_edge']).any()
        # A physical constant bias has the same RMSE regardless of the case grid.
        s=scalar(y,y+np.float32(.5),v,4.)
        assert abs(s['rmse_mps']-.5)<1e-5
        ideal=screening(y,y,masks['roof']&v)
        assert ideal['top10_recall']==1 and abs(ideal['top10_regret_mps'])<1e-8
    w=height_weights([1.001,1.002,1.003,1.10,1.30,1.49]);assert abs(w.sum()-1)<1e-12
    assert w[:3].sum()<.15  # near-roof dense samples cannot dominate merely by count
    print('PASS: 56 job memberships, 5 family folds, 20 actual grids, all 919 strict heights, reference/mask/metric contracts')

if __name__=='__main__':main()
