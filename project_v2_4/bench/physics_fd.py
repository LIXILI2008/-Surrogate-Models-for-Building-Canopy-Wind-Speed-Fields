"""Identical finite-difference continuity diagnostic for every model and CFD labels."""
import numpy as np
from scipy.ndimage import binary_erosion


def continuity(volume,valid,z_ratios,bounds,h):
    # volume layout (z,y,x,3), velocities nondimensionalized by U(H).
    if len(z_ratios)<3:return dict(rms=None,n=0)
    size=volume.shape[1];x=np.linspace(bounds[0]/h,bounds[2]/h,size);y=np.linspace(bounds[1]/h,bounds[3]/h,size)
    gx=np.gradient(volume[...,0],x,axis=2,edge_order=2)
    gy=np.gradient(volume[...,1],y,axis=1,edge_order=2)
    gz=np.gradient(volume[...,2],np.array(z_ratios),axis=0,edge_order=2)
    # Every stencil neighbour must be supported; omit outer faces and roof boundary.
    interior=binary_erosion(valid,np.ones((3,3,3),bool),border_value=0)
    values=(gx+gy+gz)[interior]
    return dict(rms=float(np.sqrt(np.mean(values.astype('float64')**2))) if len(values) else None,n=len(values),
                units='dimensionless divergence: H/U(H) * div(u)',
                method='same centered finite differences on held-out slice stack for all architectures and labels')
