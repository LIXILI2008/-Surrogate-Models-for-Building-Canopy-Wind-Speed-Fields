"""Independent continuation copies; never replace the required iteration-700 reference."""
from __future__ import annotations
import datetime as dt
from pathlib import Path
import sys
import traceback
import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from study import ART,WORK,EXTERNAL,read,write,digest
from analysis_pipeline import grid,region_masks,scalar,screening
sys.path.insert(0,str(WORK/'cfd_recovery'))
from export_high_buildings import configuration,residual_file,FIELDS,validate_ascii
from continue_priority_case import parse_residuals,assessment

OUT=ART/'cfd_sensitivity'


def select():
    meta=read(EXTERNAL/'manifest.json');geos=meta['geometry'];items=list(geos.items())
    names=[min(items,key=lambda q:q[1]['dims'][2])[0],max(items,key=lambda q:q[1]['dims'][2])[0],
           max(items,key=lambda q:q[1]['dims'][1])[0],max([q for q in items if q[1]['dims'][2]>50],key=lambda q:q[1]['dims'][0])[0]]
    passed=sorted(n for n,g in items if g['dims'][2]>50 and g['residual_criteria_passed'] is True)
    names+=passed[:1];names=list(dict.fromkeys(names));records=[]
    for name in names:
        g=geos[name];rr=[r for r in meta['records'] if r['building']==name]
        chosen=sorted({min(range(len(rr)),key=lambda i:abs(rr[i]['ratio']-q)) for q in [1.01,1.05,1.10,1.25,1.45]},key=lambda i:rr[i]['z'])
        records.append(dict(building=name,**g,planes=[rr[i] for i in chosen]))
    p=dict(selection='Before model ranking: min H, max H, max L, max high-group W, and one residual-passed high case.',
           additional_iterations=[100,300,500],building_count=len(records),buildings=records,
           original_reference_retained=True,mesh_sensitivity='separate experiment; iteration continuation is not mesh independence')
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'protocol.json'
    if path.exists():assert read(path)==p
    else:write(path,p)
    return p


def interpolate(path,row):
    xy,_=grid(row['dims']);query=xy+np.asarray(row['center'])
    with np.load(path) as a:
        coords=np.column_stack([a['x-coordinate'],a['y-coordinate']]).astype(float);v=a['velocity-magnitude'].astype(float)
    unique,inv=np.unique(np.round(coords,7),axis=0,return_inverse=True);values=np.bincount(inv,weights=v)/np.bincount(inv)
    target=np.asarray(LinearNDInterpolator(unique,values)(query),dtype='float32')
    return xy,target,np.isfinite(target)


def export_compare(solver,entry,folder,step):
    directory=folder/f'delta_{step}';directory.mkdir(exist_ok=True);results=[]
    for row in entry['planes']:
        z=row['z'];surface=f'suzhou_sensitivity_z{z:g}';surfaces=solver.settings.results.surfaces.iso_surface
        if surface not in surfaces.get_object_names():
            surfaces.create(name=surface);surfaces[surface].set_state({'field':'z-coordinate','iso_values':[float(z)]})
        csv=directory/f'z{z:g}.csv';npz=csv.with_suffix('.npz')
        if not npz.exists():
            solver.file.export.ascii(file_name=str(csv),surface_name_list=[surface],delimiter='comma',cell_func_domain=FIELDS,location='node')
            validate_ascii(csv,z)
        xy,new,nv=interpolate(npz,row)
        with np.load(EXTERNAL/row['cache']) as a:old=a['speed_mps'].copy();ov=a['valid'].copy()
        masks=region_masks(xy,row['dims']);valid=ov&nv;uin=float(3*(z/10)**.4)
        scores={name:scalar(old,new,valid&mask,uin) for name,mask in masks.items()}
        results.append(dict(z_m=z,ratio=z/row['dims'][2],regions=scores,
            old_candidate_quality_on_continued_field=screening(new,old,valid&masks['roof']),
            source_plane_sha256=row['source_sha256'],continued_plane_sha256=digest(npz),valid_fraction=float(valid.mean())))
    write(directory/'field_change.json',results)
    return results


def run():
    import ansys.fluent.core as pyfluent
    protocol=select();status=dict(state='RUNNING',completed=[],originals_modified=False,
        primary_iteration700_labels_replaced=False,physical_validation='not_established')
    write(OUT/'status.json',status)
    try:
        for entry in protocol['buildings']:
            label=entry['label'];folder=OUT/label;folder.mkdir(exist_ok=True)
            if (folder/'result.json').exists():status['completed'].append(read(folder/'result.json'));continue
            solver=None;hashes={k:digest(entry[k]) for k in ['case','data']}
            try:
                status.update(current=label,phase='loading');write(OUT/'status.json',status)
                solver=pyfluent.launch_fluent(product_version='24.1',dimension=3,precision='double',processor_count=2,
                    mode='solver',ui_mode='hidden_gui',cwd=str(folder),start_timeout=120,cleanup_on_exit=True)
                solver.file.read_case(file_name=entry['case']);solver.file.read_data(file_name=entry['data'])
                before=configuration(solver);write(folder/'configuration_before.json',before)
                activity=solver.solution.calculation_activity.get_state();write(folder/'calculation_activity.json',activity)
                if activity.get('execute_commands') or activity.get('solution_animations') or activity.get('case_modification',{}).get('enabled'):
                    raise ValueError('Active saved iteration actions require isolation before continuation')
                solver.file.auto_save.root_name=str(folder/'autosave')
                first=folder/f'residual_start_{dt.datetime.now():%Y%m%d_%H%M%S}.xy';residual_file(solver,first)
                initial=assessment(parse_residuals(first),before['residual_equations']);start=int(initial['iteration']);current=start
                write(folder/'initial_assessment.json',initial)
                if entry['label'].startswith('U'):assert start==700
                # Residual-stop flags alone are disabled to compare fixed additional counts.
                equations=solver.solution.monitor.residual.equations
                for n,v in before['residual_equations'].items():equations[n].check_convergence=False
                snapshots=[]
                for extra in protocol['additional_iterations']:
                    target=start+extra
                    while current<target:
                        count=min(50,target-current);solver.solution.run_calculation.iterate(iter_count=count)
                        path=folder/f'residual_iter{current+count}_{dt.datetime.now():%H%M%S}.xy';residual_file(solver,path)
                        state=assessment(parse_residuals(path),before['residual_equations'])
                        assert int(state['iteration'])==current+count;current=int(state['iteration'])
                        status.update(current=label,phase='iterating',current_iteration=current,additional=current-start,
                                      residual_criteria_passed=state['all_passed']);write(OUT/'status.json',status)
                        print('CFD sensitivity',label,current,flush=True)
                    field_changes=export_compare(solver,entry,folder,extra)
                    for n,v in before['residual_equations'].items():equations[n].check_convergence=v['check_convergence']
                    assert configuration(solver)==before
                    checkpoint=folder/f'continued_iter{current}.cas.h5'
                    # Incomplete prior attempts may be revisited, but never an input case/data pair.
                    if not checkpoint.exists():solver.file.write_case_data(file_name=str(checkpoint))
                    snapshots.append(dict(iteration=current,additional_iterations=extra,residual_assessment=state,field_changes=field_changes))
                    write(folder/'snapshots.json',snapshots)
                    for n,v in before['residual_equations'].items():equations[n].check_convergence=False
                for n,v in before['residual_equations'].items():equations[n].check_convergence=v['check_convergence']
                assert configuration(solver)==before
                assert hashes=={k:digest(entry[k]) for k in ['case','data']}
                result=dict(building=entry['building'],label=label,source_sha256=hashes,initial_iteration=start,
                    final_iteration=current,snapshots=snapshots,originals_modified=False,settings_unchanged=True,
                    physical_validation='iteration sensitivity only; mesh independence remains unverified')
                write(folder/'result.json',result);status['completed'].append(result);write(OUT/'status.json',status)
                print('CFD BUILDING COMPLETE',label,flush=True)
            finally:
                if solver is not None:solver.exit()
        status.update(state='ITERATION_SENSITIVITY_COMPLETE',current=None);write(OUT/'status.json',status)
    except BaseException:
        status.update(state='FAILED',error=traceback.format_exc());write(OUT/'status.json',status);raise


if __name__=='__main__':
    from bench.runner import process_alive
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'runner.lock'
    if lock.exists() and process_alive(read(lock)['pid']):raise RuntimeError('CFD sensitivity queue already active')
    write(lock,dict(pid=os.getpid()))
    try:run()
    finally:
        if lock.exists() and read(lock)['pid']==os.getpid():lock.unlink()
