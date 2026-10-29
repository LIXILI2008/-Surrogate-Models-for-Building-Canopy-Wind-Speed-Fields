from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from study import ART,read,write,digest

OUT=ART/'three_followups_v1'

def main():
    order=[s['directory'] for s in read(ART/'prospective_five_v1/protocol.json')['cases']]
    r=read(OUT/'adapted_new5/observations.json');fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    x=np.arange(5)
    for i,p in enumerate(['hard','random','diverse']):
        delta=[np.mean([a['after']-a['before'] for a in r if a['policy']==p and a['building']==n]) for n in order]
        ax.bar(x+(i-1)*.24,delta,.23,label=p)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,['P01','P02','P03','P04','P05']);ax.set(ylabel='Adapted minus original RMSE (m/s)',title='Exploratory K=8 adaptation: negative = improvement')
    ax.legend(title='Support acquisition');fig.savefig(OUT/'adaptation_new5.png',dpi=200);plt.close(fig)
    r=read(OUT/'label_source/controlled.json')['rows'];fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for i,p in enumerate(['exact_50band','calibrated_lut']):
        vals=[np.mean([a['unsaturated']['rmse'] for a in r if a['route']==p and a['building']==n]) for n in order]
        ax.bar(x+(i-.5)*.33,vals,.32,label=p)
    ax.set_xticks(x,['P01','P02','P03','P04','P05']);ax.set(ylabel='Image inversion RMSE (m/s)',title='Controlled same-field PNG inversion (unsaturated pixels)');ax.legend()
    fig.savefig(OUT/'controlled_label_error.png',dpi=200);plt.close(fig)
    rows=read(OUT/'negative_transfer/rows.json')['summary'];bands=['1-1.02','1.02-1.1','1.1-1.25','1.25-1.5'];regions=['full','roof','wake','windward','side']
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained');data=[]
    for group in ('low','high'):
        a=np.array([[next(r['mean_delta'] for r in rows if r['kind']=='main_grouped_cv' and r['policy']=='hard' and r['k']==8 and r['condition']=='group' and r['value']==group and r['band']==b and r['region']==reg) for reg in regions] for b in bands]);data.append(a)
    limit=max(abs(a).max() for a in data)
    for ax,a,g in zip(axs,data,['Low: 10 buildings','High: 10 buildings']):
        im=ax.imshow(a,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto');ax.set_xticks(range(5),regions);ax.set_yticks(range(4),bands);ax.set(title=g,ylabel='z/H band')
        for i in range(4):
            for j in range(5):ax.text(j,i,f'{a[i,j]:+.3f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=axs,label='RMSE change (m/s); negative = improvement');fig.savefig(OUT/'negative_transfer_bands.png',dpi=200);plt.close(fig)
    write(OUT/'completion.json',dict(state='COMPLETE',adapted_checkpoints=45,controlled_cases=25,historical_matched_planes=89,
        caveat='Historical low-group source-domain effect remains unidentifiable without native matches; prospective adaptation is exploratory.',
        source_hashes={str(p.relative_to(OUT)):digest(p) for p in [OUT/'adapted_new5/protocol.json',OUT/'adapted_new5/observations.json',OUT/'label_source/controlled.json',OUT/'label_source/historical.json',OUT/'negative_transfer/rows.json']}))

if __name__=='__main__':main()
