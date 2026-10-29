"""Dataset gate: SI scaling, geometry masks, source identity, split isolation."""
import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from .common import ROOT,read,write,digest
from .labels import load_label

def audit(dataset):
    dataset=Path(dataset);m=read(dataset/'manifest.json');points=np.load(dataset/'points.npy',mmap_mode='r')
    assert digest(dataset/'points.npy')==m['points_sha256']
    oldpath=ROOT.parent/'project_v2/artifacts/dataset';old=read(oldpath/'manifest.json')
    oldpoints=np.load(oldpath/'points.npy',mmap_mode='r');oldlow=[i for i,r in enumerate(old['records']) if r['dims'][2]<=50]
    assert np.array_equal(points[:len(oldlow)],oldpoints[oldlow])
    sets=[set(m['splits'][s]) for s in ('train','validation','test')]
    assert not any(a&b for i,a in enumerate(sets) for b in sets[i+1:])
    counts={};direct=[];rng=np.random.default_rng(41)
    for i,r in enumerate(m['records']):
        assert r['building'] in m['splits'][r['split']]
        if r.get('label_domain')!='high_fluent_numeric':continue
        name=r['building'];w,l,h=r['dims'];z=r['z'];counts.setdefault(name,[]).append(z)
        xy,y,fluid=load_label(m,r,dataset)
        with np.load(dataset/r['cache']) as d:
            assert np.allclose(d['speed_mps'][d['valid']],(d['inlet_mps']+d['residual_mps'])[d['valid']],atol=2e-6)
            assert np.allclose(d['inlet_mps'],3*(z/10)**.4)
            assert not d['solid_mask'].any()
            if abs(z-h)<1e-6:
                assert np.array_equal(d['roof_boundary'],d['projection'])
                assert (y[d['roof_boundary']]==0).all() and not fluid[d['roof_boundary']].any()
            else:
                assert not d['roof_boundary'].any()
                assert np.array_equal(fluid[d['projection']],d['valid'][d['projection']])
            for field in ('sdf_m','sdf3_m','velocity_mps','nearest_cfd_node_distance_m'):
                assert np.isfinite(d[field][d['valid']]).all()
        p=np.asarray(points[i]);step=xy[0,1,0]-xy[0,0,0];dy=xy[1,0,1]-xy[0,0,1]
        cc=np.rint((p[:,0]-xy[0,0,0])/step).astype(int);rr=np.rint((p[:,1]-xy[0,0,1])/dy).astype(int)
        assert ((cc>=0)&(cc<512)&(rr>=0)&(rr<512)).all()
        assert fluid[rr,cc].all() and np.allclose(p[:,2]*13,y[rr,cc],atol=2e-6)
        if z in (h,h+1,1.5*h):
            assert digest(r['source'])==r['source_sha256']
            with np.load(r['source']) as d:
                center=m['high_grid_geometry'][name]['origin_xy_fluent_m']
                native_xy=np.stack((d['x-coordinate']-center[0],d['y-coordinate']-center[1]),-1)
                ids=rng.choice(len(p),128,replace=False)
                actual=LinearNDInterpolator(native_xy,d['velocity-magnitude'])(p[ids,:2])
                error=float(np.max(abs(actual-p[ids,2]*13)))
                assert error<2e-4,(name,z,error)
                direct.append(dict(building=name,z=z,max_scalar_interpolation_error_mps=error))
    for n,zs in counts.items():
        h=m['dims'][m['names'].index(n)][2]
        assert np.array_equal(zs,np.arange(h,1.5*h+1))
    assert len(counts)==5 and sum(map(len,counts.values()))==220
    result=dict(state='PASSED',records=len(m['records']),low_records_preserved=len(oldlow),high_slices=220,
        per_building={k:len(v) for k,v in counts.items()},direct_scalar_checks=direct,manifest_sha256=digest(dataset/'manifest.json'))
    write(dataset/'audit.json',result);print(result,flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);args=ap.parse_args();audit(args.dataset)
