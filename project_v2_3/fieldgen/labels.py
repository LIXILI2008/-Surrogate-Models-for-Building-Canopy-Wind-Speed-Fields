"""Resolve physical labels by provenance; never reinterpret an SI grid as RGB cache."""
from pathlib import Path
import numpy as np
from .common import digest

def load_label(meta,row,dataset):
    numeric=row.get('label_domain')=='high_fluent_numeric'
    path=(Path(dataset) if numeric else Path(meta['source_cache']))/row['cache']
    if digest(path)!=row['cache_sha256']:raise ValueError(f'Label hash changed: {path}')
    with np.load(path) as data:
        if numeric:
            xy=np.stack((data['x'],data['y']),-1).astype('float32')
            y=data['speed_mps'].astype('float32');valid=data['fluid_mask'].copy()
        else:
            y=data['target'][...,0].astype('float32')*13;valid=data['valid'].copy()
            g=meta['geometry'][row['building']];yy,xx=np.indices(y.shape)
            xy=np.stack(((xx-g['cx'])*g['dx'],(yy-g['cy'])*g['dy']),-1).astype('float32')
    w,l,h=row['dims'];roof=(abs(xy[...,0])<=l/2)&(abs(xy[...,1])<=w/2)&(abs(row['z']-h)<1e-6)
    valid &= ~roof
    if not valid.any() or not np.isfinite(y[valid]).all():raise ValueError('Invalid fluid label')
    return xy,y,valid
