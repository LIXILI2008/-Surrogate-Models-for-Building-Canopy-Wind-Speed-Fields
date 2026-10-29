"""Read-only inference parity audit against frozen six-method predictions."""
import numpy as np
from study import ART, METHODS, SEEDS, read, write, digest
from analysis_pipeline import Labels, load_model, infer

def main():
    p=read(ART/'protocol.json');labels=Labels('production',p['external_buildings'])
    chosen=[0,len(labels.rows)-1];results=[]
    for method in METHODS:
        key=f'production/{method}/seed_{SEEDS[0]}'
        model,_,device=load_model(dict(method=method,key=key,train=p['training_buildings'],test=p['external_buildings']))
        import torch
        torch.backends.cuda.matmul.allow_tf32=False
        saved=np.load(ART/key/'predictions.npy',mmap_mode='r')
        for i in chosen:
            row=labels.rows[i];xy,bounds,_,_,_=labels.get(row)
            pred=infer(model,method,xy,row['dims'],row['z'],bounds,device)
            delta=float(np.max(np.abs(pred-saved[i])))
            assert pred.shape==(128,128) and np.isfinite(pred).all()
            assert delta<1e-4,(method,delta)
            results.append(dict(method=method,building=row['building'],z=row['z'],maximum_absolute_difference_mps=delta,checkpoint_sha256=digest(ART/key/'best.pt')))
        print(method,'parity passed',flush=True)
    write(ART/'unified_interface_audit.json',dict(status='PASSED',cases=results,scope='Two slices per method, first production seed; exact frozen-grid inference parity, not new accuracy evidence'))

if __name__=='__main__':main()
