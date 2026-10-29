from pathlib import Path
import json,subprocess,sys,os,time
import numpy as np
W=Path(__file__).resolve().parents[2]; V=W/'project_v2_5'; cfg=json.loads((V/'config.json').read_text()); out=V/'artifacts/migration_check_20261014';out.mkdir(parents=True,exist_ok=True)
# Original entry point has no results override. Bind only ART before importing it.
bridge=out/'original_entry.py'
bridge.write_text("import sys\nfrom pathlib import Path\nsys.path.insert(0,"+repr(str(W/'project_suzhou_validation'))+")\nimport study\nstudy.ART=Path("+repr(cfg['reference_results'])+")\nimport research\nresearch.main()\n")
rows=[]
for model,old in cfg['models'].items():
 for label,dims,z in [('low',[18,18,18],18.9),('high',[25,40,90],94.5)]:
  row={'model':model,'case':label,'dims_WLH':dims,'z':z,'seed':20261001}
  for version in ['original','v25']:
   target=out/f'{model}_{label}_{version}'
   cmd=[sys.executable,str(bridge if version=='original' else V/'v25.py'),'predict','--method',old if version=='original' else model,'--dims',*map(str,dims),'--z',str(z),'--output',str(target)]
   env=dict(os.environ,PYTHONIOENCODING='utf-8')
   result=subprocess.run(cmd,cwd=W,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
   (out/f'{model}_{label}_{version}.log').write_text(result.stdout,encoding='utf-8')
   if result.returncode: raise RuntimeError(f'{model} {label} {version}: {result.stdout}')
  a=out/f'{model}_{label}_original';b=out/f'{model}_{label}_v25'
  with np.load(a/'field.npz') as x,np.load(b/'field.npz') as y:
   row['arrays']={k:dict(exact=bool(np.array_equal(x[k],y[k])),max_absolute_difference=float(np.max(np.abs(x[k]-y[k]))),finite=bool(np.isfinite(x[k]).all() and np.isfinite(y[k]).all())) for k in x.files}
  pa=json.loads((a/'provenance.json').read_text());pb=json.loads((b/'provenance.json').read_text())
  row['checkpoint_identical']=pa['checkpoint_sha256']==pb['checkpoint_sha256']
  row['provenance_identical']=all(pb.get(k)==v for k,v in pa.items())
  row['passed']=row['checkpoint_identical'] and row['provenance_identical'] and all(v['exact'] and v['finite'] for v in row['arrays'].values())
  rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(model,label,row['passed'],flush=True)
(out/'REPORT.md').write_text('# V2.5入口迁移一致性核验\n\n同一历史权重、种子20261001，共六配置×高低两个案例，分别调用原research.main与V2.5命令行。原入口仅绑定相同结果目录；未更改推理逻辑。\n\n|配置|案例|风速最大绝对差m/s|逐元素一致|权重相同|\n|---|---|---:|---|---|\n'+''.join(f"|{r['model']}|{r['case']}|{r['arrays']['speed_mps']['max_absolute_difference']}|{r['passed']}|{r['checkpoint_identical']}|\n" for r in rows)+'\n范围：预测入口及现有权重兼容；没有重新训练，没有验证训练/适配全流程，没有重新认证原指标或CFD。\n',encoding='utf-8')
