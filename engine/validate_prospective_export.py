"""Verify exported scientific data and its shared V2.3/V2.4 reader contract."""
from pathlib import Path
import argparse, json, time, os, ctypes, hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parent/'artifacts/prospective_five_v1/export'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def alive(pid):
    h=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    if not h:return False
    code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(h,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(h)
    return code.value==259
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--wait-pid',type=int);a=ap.parse_args()
    if a.wait_pid:
        while alive(a.wait_pid):time.sleep(10)
    m=read(ROOT/'dataset/manifest.json');specs=read(ROOT.parent/'protocol.json')['cases'];report={}
    from analysis_pipeline import Labels
    ds=Labels('production',m['names'],external_root=ROOT/'dataset')
    for name in m['names']:
        spec=next(s for s in specs if s['directory']==name);h=spec['H']
        rows=[r for r in m['records'] if r['building']==name]
        expected=set(range(int(1.5*h)+1))|{round(h-1+i*.1,4) for i in range(21)}
        assert len(rows)==len(expected) and {r['z'] for r in rows}==expected
        errors=[]
        for r in rows:
            assert r.get('render_revision')=='sorted_velocity_lut_v2', 'Uncorrected colour rendering'
            cache=ROOT/'dataset'/r['cache'];image=Path(r['image']);source=Path(r['source'])
            assert all(digest(p)==s for p,s in [(cache,r['cache_sha256']),(image,r['image_sha256']),(source,r['source_sha256'])])
            with np.load(cache) as x:
                v=x['valid'];speed=x['speed_mps'];target=x['target'];uin=float(x['uin_mps'])
                assert speed.shape==(128,128) and v.dtype==bool and target.shape==(7,128,128)
                assert v.any() and np.isfinite(speed[v]).all() and np.isfinite(target).all()
                if uin>0:
                    err=float(abs((target[0][v]+1)*uin-speed[v]).max());assert err<1e-4;errors.append(err)
                else:assert not x['normalized_valid'].any()
            iw,ih=Image.open(image).size;x0,y0,x1,y1=r['image_bounds']
            assert abs((x1-x0)/iw-(y1-y0)/ih)<1e-9
        complete=read(ROOT/spec['id']/'complete.json')
        assert complete['source_hashes_unchanged'] and complete['solver_configuration_unchanged']
        for r in ds.rows:
            if r['building']==name:
                xy,b,y,v,d=ds.get(r);assert y.shape==v.shape==(128,128) and xy.shape==(128,128,2)
        report[name]=dict(slices=len(rows),roof_evaluation_slices=sum(r['roof_evaluation_eligible'] for r in rows),
            max_normalization_roundtrip_error_mps=max(errors),all_hashes_match=True,physical_image_aspect_preserved=True)
    result=dict(state='PASSED_FOR_COMPLETED_BUILDINGS',buildings=report,complete_buildings=len(report),expected_buildings=5,
                convergence_certified=False,reader='analysis_pipeline.Labels production with external_root',pid=os.getpid())
    (ROOT/'dataset_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
