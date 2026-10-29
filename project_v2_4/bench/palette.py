import csv
import numpy as np
from .common import ROOT,digest

HASHES={'real_lut_0_6.csv':'cc47d0c5aa9ecd88b2bfd594f96a9328f3280a956e49da5a97aeed8dc253b86d',
        'real_lut_0_13.csv':'3ff9c7fdfb0caade70f6fb465a5d76a0e47e10fc18dcecebb27992741811dad7'}


def render(speed,h):
    name='real_lut_0_6.csv' if h<=50 else 'real_lut_0_13.csv';path=ROOT/'configs'/name
    assert digest(path)==HASHES[name]
    with path.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    x=np.array([float(r['value']) for r in rows]);rgb=np.array([[float(r[k]) for k in ('r','g','b')] for r in rows])
    order=np.argsort(x)
    return np.stack([np.interp(speed,x[order],rgb[order,c]) for c in range(3)],-1).round().clip(0,255).astype('uint8')
