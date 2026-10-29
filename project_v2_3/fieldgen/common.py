import hashlib
import json
from pathlib import Path
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
    return h.hexdigest()


def check_query(dims,z,bounds):
    d=np.asarray(dims,float)
    if d.shape!=(3,) or not np.isfinite(d).all() or not np.isfinite(z):raise ValueError('W,L,H,z must be finite')
    if np.any(d<np.asarray(bounds['min'])) or np.any(d>np.asarray(bounds['max'])):
        raise ValueError(f"W,L,H outside supported dataset bounds: {bounds}")
    if not d[2]<=z<=1.5*d[2]:raise ValueError('Required height range: H <= z <= 1.5H')
    return d
