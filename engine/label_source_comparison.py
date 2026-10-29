"""Separate historical label disagreement from controlled PNG inversion error."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy.interpolate import LinearNDInterpolator,RegularGridInterpolator
from scipy.spatial import cKDTree
from study import ART,WORK,read,write,digest

OUT=ART/'three_followups_v1/label_source'

def score(a,b,v):
    e=b[v]-a[v]
    return dict(n=int(v.sum()),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(e.mean())) if v.any() else dict(n=0,rmse=None,mae=None,bias=None)

def main():
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'status.json',dict(state='RUNNING'))
    root=ART/'prospective_five_v1/export';meta=read(root/'dataset/manifest.json');controlled=[]
    if (OUT/'controlled.json').exists():controlled=read(OUT/'controlled.json')['rows']
    for name in ([] if controlled else meta['names']):
        available=[r for r in meta['records'] if r['building']==name and r['roof_evaluation_eligible']]
        chosen={min(range(len(available)),key=lambda i:abs(available[i]['ratio']-q)) for q in (1.01,1.05,1.1,1.25,1.45)}
        for i in sorted(chosen):
            r=available[i];assert digest(r['image'])==r['image_sha256'] and digest(r['source'])==r['source_sha256']
            img=np.asarray(Image.open(r['image']).convert('RGB'));ih,iw=img.shape[:2];iy=np.arange(0,ih,4);ix=np.arange(0,iw,4);pixels=img[np.ix_(iy,ix)]
            x0,y0,x1,y1=r['image_bounds'];x=x0+(ix+.5)*(x1-x0)/iw;y=y1-(iy+.5)*(y1-y0)/ih;xx,yy=np.meshgrid(x,y)
            with np.load(r['source']) as a:xy=np.column_stack([a['x-coordinate'],a['y-coordinate']]);vel=a['velocity-magnitude']
            xy,inv=np.unique(np.round(xy,7),axis=0,return_inverse=True);vel=np.bincount(inv,weights=vel)/np.bincount(inv)
            truth=LinearNDInterpolator(xy,vel)(xx,yy);lim=6 if r['dims'][2]<=50 else 13
            lut=np.loadtxt(WORK/f'project_v2_3/configs/real_lut_0_{lim}.csv',delimiter=',',skiprows=1);lut=lut[np.argsort(lut[:,3])]
            palette_values=(np.arange(50)+.5)*lim/50
            palette=np.stack([np.interp(palette_values,lut[:,3],lut[:,k]) for k in range(3)],-1).astype('uint8')
            for route,col,val in [('exact_50band',palette,palette_values),('calibrated_lut',lut[:,:3],lut[:,3])]:
                dist,idx=cKDTree(col.astype(float)).query(pixels.reshape(-1,3));recovered=val[idx].reshape(truth.shape)
                valid=np.isfinite(truth)&~(pixels>250).all(-1);unsaturated=valid&(truth>=0)&(truth<lim)
                controlled.append(dict(building=name,z=r['z'],ratio=r['ratio'],route=route,limit=lim,all_valid=score(truth,recovered,valid),
                    unsaturated=score(truth,recovered,unsaturated),saturated_fraction=float((valid&~unsaturated).sum()/max(1,valid.sum())),
                    pixel_stride=4,image_sha256=r['image_sha256'],source_sha256=r['source_sha256']))
    write(OUT/'controlled.json',dict(rows=controlled,scope='Same native snapshot, registered physical pixel centres and PNG; rendering/inversion diagnostic, not independent CFD validation or historical low-label bias estimate'))
    old=read(WORK/'project_v2/artifacts/dataset/manifest.json');native_root=WORK/'project_v2_3/artifacts/dataset_verified';native=read(native_root/'manifest.json')
    lookup={(r['building'],round(r['z'],5)):r for r in native['records'] if r.get('label_domain')=='high_fluent_numeric' and 1+1e-7<r['ratio']<1.5-1e-7}
    historical=[]
    for r in old['records']:
        nr=lookup.get((r['building'],round(r['z'],5)))
        if nr is None:continue
        g=old['geometry'][r['building']];cp=Path(old['source_cache'])/r['cache'];assert digest(cp)==r['cache_sha256']
        with np.load(cp) as a:im=a['target'][...,0]*13;valid=a['valid'];shape=im.shape
        with np.load(native_root/nr['cache']) as a:nx=a['x'][0];ny=a['y'][:,0];speed=a['speed_mps'];nv=a['fluid_mask']
        # Native queries remain fixed; neither convention is selected by error.
        bounds=[max(nx.min(),r['observed_xy_bounds'][0]),max(ny.min(),r['observed_xy_bounds'][1]),min(nx.max(),r['observed_xy_bounds'][2]),min(ny.max(),r['observed_xy_bounds'][3])]
        bounds=list(map(float,bounds))
        if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:continue
        x,y=np.meshgrid(np.linspace(bounds[0],bounds[2],128),np.linspace(bounds[1],bounds[3],128));q=np.stack([y,x],-1)
        target=RegularGridInterpolator((ny,nx),speed,bounds_error=False,fill_value=np.nan)(q)
        supported=RegularGridInterpolator((ny,nx),nv.astype(float),bounds_error=False,fill_value=0)(q)>.999
        ox=(np.arange(shape[1])-g['cx'])*g['dx'];oy=(np.arange(shape[0])-g['cy'])*g['dy']
        for convention,sign in [('stored_dataset_coordinates',1),('y_reflected_sensitivity',-1)]:
            oq=np.stack([sign*y,x],-1);label=RegularGridInterpolator((oy,ox),im,bounds_error=False,fill_value=np.nan)(oq)
            ov=RegularGridInterpolator((oy,ox),valid.astype(float),bounds_error=False,fill_value=0)(oq)>.999
            mask=supported&ov&np.isfinite(target)&np.isfinite(label)
            historical.append(dict(building=r['building'],z=r['z'],convention=convention,common_bounds=bounds,**score(target,label,mask),
                image_cache_sha256=r['cache_sha256'],native_cache_sha256=nr['cache_sha256'],same_solver_snapshot_proven=False))
    write(OUT/'historical.json',dict(rows=historical,scope='Exact same building/height, common physical queries and masks; solver snapshot correspondence and registration are not certified. Cannot isolate image inversion causally. Both y conventions reported, not best-picked.',
        original_low_buildings_with_native_match=0))
    lines=['# 标签来源对照','','## 受控成图—反解实验','',
        '同一原生CFD快照与已注册图像，固定5个相对高度附近的实际切片，每4像素采样；先切片平均再建筑平均。不代表历史低组标签误差。','',
        '|建筑|反解方式|未饱和RMSE m/s|未饱和MAE|','|---|---|---:|---:|']
    for n in meta['names']:
        for route in ('exact_50band','calibrated_lut'):
            rr=[r for r in controlled if r['building']==n and r['route']==route]
            lines.append(f"|{n}|{route}|{np.mean([r['unsaturated']['rmse'] for r in rr]):.5f}|{np.mean([r['unsaturated']['mae'] for r in rr]):.5f}|")
    lines+=['','## 历史图像标签与数值标签','','只选建筑及z完全匹配的记录；相同物理坐标、共同掩膜。求解时刻无法独立核实，且历史图像y方向存在约定差异；两个方向全部报告，不能择优后宣称严格验证。','',
        '|建筑|坐标约定|匹配层数|共同区域平均RMSE m/s|','|---|---|---:|---:|']
    for n in sorted({r['building'] for r in historical}):
        for c in ('stored_dataset_coordinates','y_reflected_sensitivity'):
            rr=[r for r in historical if r['building']==n and r['convention']==c and r['n']]
            lines.append(f"|{n}|{c}|{len(rr)}|{np.mean([r['rmse'] for r in rr]):.5f}|")
    lines+=['','原低组23种几何没有在本次已整理的原生数值库中获得严格配对；因此不能将历史低组误差定量归因于图像反解。后续需恢复其原始同快照切片及独立相机注册。']
    (OUT/'REPORT.md').write_text('\n'.join(lines),encoding='utf8');write(OUT/'status.json',dict(state='COMPLETE',controlled_rows=len(controlled),historical_rows=len(historical)))

if __name__=='__main__':
    try:main()
    except BaseException:
        import traceback
        write(OUT/'failure.json',dict(error=traceback.format_exc()));raise
