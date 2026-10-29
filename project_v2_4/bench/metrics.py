import numpy as np
from scipy.ndimage import binary_erosion
from skimage.metrics import structural_similarity
from .common import inlet


def binary(truth,pred):
    n=int(truth.size);tp=int((truth&pred).sum());a=int(truth.sum());b=int(pred.sum());union=a+b-tp
    return dict(n=n,truth_positive=a,pred_positive=b,intersection=tp,union=union,
                iou=tp/union if union else None,truth_fraction=a/n if n else None,
                precision=tp/b if b else None,recall=tp/a if a else None,
                false_positive_fraction=(b-tp)/n if n else None,
                truth_empty=a==0)


def scalar(y,p):
    y=np.asarray(y,dtype=np.float64);p=np.asarray(p,dtype=np.float64)
    if not len(y):return dict(n=0,rmse=None,mae=None,r2=None,bias=None)
    e=p-y;sst=float(((y-y.mean())**2).sum())
    return dict(n=len(y),rmse=float(np.sqrt(np.mean(e*e))),mae=float(abs(e).mean()),
                r2=1-float((e*e).sum())/sst if sst>1e-8 else None,bias=float(e.mean()))


def score(dtrue,dpred,valid,q,dims,z):
    u0=inlet(z);y=(dtrue+1)*u0;p=(dpred+1)*u0
    x=q[...,0]*dims[2];yy=q[...,1]*dims[2];w,l,h=dims
    # Fixed geometric ROI, independent of target and prediction. Flow is along +x.
    wake=valid & (x>=l/2) & (x<=l/2+2*h) & (abs(yy)<=max(w, .25*h))
    roof=valid & (abs(x)<=l/2+3) & (abs(yy)<=w/2+3)
    row=dict(speed=scalar(y[valid],p[valid]),disturbance=scalar(dtrue[valid],dpred[valid]),
             geometric_wake=scalar(y[wake],p[wake]),roof_neighbourhood=scalar(y[roof],p[roof]),
             iou_at_4=binary(y[valid]>=4,p[valid]>=4))
    for threshold in (.1,.2):
        key=f'{threshold:g}'
        row['deficit_'+key]=binary(dtrue[valid]<=-threshold,dpred[valid]<=-threshold)
        row['acceleration_'+key]=binary(dtrue[valid]>=threshold,dpred[valid]>=threshold)
        row['wake_deficit_'+key]=binary(dtrue[wake]<=-threshold,dpred[wake]<=-threshold)
    centres=binary_erosion(valid,np.ones((11,11),bool),border_value=0)
    for key,a,b,extent in [('ssim_speed',y,p,6 if h<=50 else 13),('ssim_disturbance',dtrue,dpred,2.)]:
        if centres.any():
            _,m=structural_similarity(np.where(valid,a,0),np.where(valid,b,0),data_range=extent,
                    gaussian_weights=True,sigma=1.5,use_sample_covariance=False,full=True)
            row[key]=float(m[centres].mean())
        else:row[key]=None
    return row
