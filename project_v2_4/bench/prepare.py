"""Derive dimensionless targets without changing any original SI/RGB labels."""
import argparse
import sys
import re
import traceback
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.interpolate import LinearNDInterpolator
from .common import ROOT,WORKSPACE,SOURCE,read,write,digest,inlet
from .geometry import grid,masks_numpy


def physics_settings(meta):
    sys.path.insert(0,str(WORKSPACE/'artifacts/cfd_recovery/dependencies'))
    import h5py
    rows=[]
    expected={'kecmu':.09,'kec1':1.44,'kec2':1.92,'kesigk':1.,'kesige':1.3}
    for name in meta['high_grid_geometry']:
        completed=read(WORKSPACE/'artifacts/cfd_recovery/high_numeric_final'/name/'complete.json')
        configpath=WORKSPACE/'artifacts/cfd_recovery/convergence_audit'/name/'configuration.json'
        config=read(configpath)['setup']
        assert config['models']['viscous']['model']=='k-epsilon'
        assert config['models']['viscous']['k_epsilon_model']=='standard'
        assert not config['general']['operating_conditions']['gravity']['enable']
        air=config['materials']['fluid']['air']
        rho=air['density']['value'];mu=air['viscosity']['value']
        with h5py.File(completed['source_case'],'r') as f:
            raw=f['settings/Rampant Variables'][0].decode()
        constants={key:float(re.search(r'\('+key+r' ([^)]+)\)',raw)[1]) for key in expected}
        assert constants==expected and abs(rho-1.225)<1e-8 and abs(mu-1.7894e-5)<1e-10
        rows.append(dict(building=name,constants=constants,rho=rho,mu=mu,
                         source_case=completed['source_case'],configuration_sha256=digest(configpath)))
    return dict(rho=1.225,mu=1.7894e-5,constants=expected,evidence=rows)


def load_original(meta,row):
    numeric=row['label_domain']=='high_fluent_numeric'
    path=(SOURCE if numeric else __import__('pathlib').Path(meta['source_cache']))/row['cache']
    assert digest(path)==row['cache_sha256'],path
    with np.load(path) as a:
        if numeric:
            x=a['x'];y=a['y'];speed=a['speed_mps'].copy();valid=a['fluid_mask'].copy()
        else:
            speed=a['target'][...,0].astype('float32')*13;valid=a['valid'].copy()
            g=meta['geometry'][row['building']];yy,xx=np.indices(speed.shape)
            x=(xx-g['cx'])*g['dx'];y=(yy-g['cy'])*g['dy']
    w,l,h=row['dims'];roof=(abs(x)<=l/2)&(abs(y)<=w/2)&(abs(row['z']-h)<1e-6)
    valid &= ~roof
    return x,y,speed,valid


def resample(x,y,value,valid,q,h):
    coords=np.stack(((q[...,1]*h-y[0,0])/(y[1,0]-y[0,0]),
                     (q[...,0]*h-x[0,0])/(x[0,1]-x[0,0])))
    support=map_coordinates(valid.astype('float32'),coords,order=1,mode='constant',cval=0)
    weighted=map_coordinates(np.where(valid,value,0),coords,order=1,mode='constant',cval=0)
    return weighted/np.maximum(support,1e-8),support>.999


def main(size=128):
    out=ROOT/'artifacts'/f'dataset_{size}';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():
        saved=read(out/'manifest.json');assert saved['source_manifest_sha256']==digest(SOURCE/'manifest.json')
        for f,v in saved['array_sha256'].items():assert digest(out/f)==v
        return
    meta=read(SOURCE/'manifest.json');physics=physics_settings(meta)
    rows=[];n=len(meta['records'])
    target=np.lib.format.open_memmap(out/'targets.npy',mode='w+',dtype='float32',shape=(n,7,size,size))
    valid=np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype='bool',shape=(n,size,size))
    auxvalid=np.lib.format.open_memmap(out/'aux_valid.npy',mode='w+',dtype='bool',shape=(n,size,size))
    checks=[]
    for i,row in enumerate(meta['records']):
        x,y,speed,ok=load_original(meta,row);w,l,h=row['dims'];z=row['z'];u0=inlet(z);uh=inlet(h)
        bounds=[float(x[0,0]),float(y[0,0]),float(x[-1,-1]),float(y[-1,-1])]
        q=grid(bounds,row['dims'],z,size);_,roof,fluid=masks_numpy(q,row['dims'])
        sampled,supported=resample(x,y,speed,ok,q,h);supported &= fluid
        t=np.zeros((size,size,7),np.float32);t[...,0]=np.where(supported,sampled/u0-1,0)
        aok=np.zeros_like(supported)
        if row['label_domain']=='high_fluent_numeric':
            assert digest(row['source'])==row['source_sha256']
            with np.load(row['source']) as native:
                center=meta['high_grid_geometry'][row['building']]['origin_xy_fluent_m']
                xy=np.stack((native['x-coordinate']-center[0],native['y-coordinate']-center[1]),-1)
                fields=np.stack([native[k] for k in ['x-velocity','y-velocity','z-velocity','pressure','turb-kinetic-energy','turb-diss-rate']],-1)
            values=LinearNDInterpolator(xy,fields)(q[...,:2]*h)
            aok=supported & np.isfinite(values).all(-1) & (values[...,4]>=0) & (values[...,5]>0)
            scales=np.array([uh,uh,uh,physics['rho']*uh**2,uh**2,uh**3/h])
            t[...,1:]=np.where(aok[...,None],values/scales,0)
        t[roof,0]=-1 # prescribed stationary roof; never a fluid metric pixel
        target[i]=t.transpose(2,0,1);valid[i]=supported;auxvalid[i]=aok
        assert np.isfinite(t).all() and supported.any()
        err=float(np.max(abs((t[...,0][supported]+1)*u0-sampled[supported])))
        assert err<1e-5
        checks.append(err)
        rows.append(dict(row,bounds=bounds,span=[bounds[2]-bounds[0],bounds[3]-bounds[1]],index=i))
        if i%20==0 or i==n-1:
            write(out/'status.json',dict(state='BUILDING',records=i+1,total=n,current=row['building']))
            print('prepare',i+1,n,flush=True)
    for a in (target,valid,auxvalid):a.flush()
    del target,valid,auxvalid
    write(out/'manifest.json',dict(schema=4,size=size,records=rows,names=meta['names'],dims=meta['dims'],bounds=meta['bounds'],
        source_manifest_sha256=digest(SOURCE/'manifest.json'),physics=physics,
        array_sha256={f:digest(out/f) for f in ['targets.npy','valid.npy','aux_valid.npy']},
        targets=['D=speed/Uin(z)-1','u/U(H)','v/U(H)','w/U(H)','p/(rho*U(H)^2)','k/U(H)^2','epsilon*H/U(H)^3'],
        thresholds=[.1,.2],negative_class='D <= -threshold',positive_class='D >= threshold',
        label_policy='SI magnitude preserved by reversible transform; auxiliary numeric labels only for high buildings. No pseudo-vector labels for low buildings.',
        max_roundtrip_error_mps=max(checks),resampling='128x128 bilinear valid-only scalar interpolation; support must exceed .999. Not directly comparable to original 512-grid scores.'))
    write(out/'status.json',dict(state='COMPLETE',records=n,roundtrip_max_mps=max(checks)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--size',type=int,default=128);a=p.parse_args()
    try:main(a.size)
    except BaseException:
        write(ROOT/'artifacts'/f'dataset_{a.size}'/'status.json',dict(state='FAILED',error=traceback.format_exc()));raise
