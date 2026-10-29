"""Steady incompressible 3-D standard k-epsilon RANS residuals in x/H, U/U(H).

Interior residual regularization with observed k/epsilon/pressure; not a replacement
for Fluent wall functions and not an independently well-posed CFD boundary solver.
"""
import torch


def grad(y,x):
    if not y.requires_grad:return x*0
    g=torch.autograd.grad(y.sum(),x,create_graph=True,retain_graph=True,allow_unused=True)[0]
    return x*0 if g is None else g+x*0


def residuals(output,q,dims,rho=1.225,mu=1.7894e-5):
    # q gradients are dimensionless spatial derivatives, holding geometry fixed.
    u=output[:,1:4];p=output[:,4];k=output[:,5];eps=output[:,6]
    h=dims[:,2];uh=3*(h/10)**.4;nu=(mu/rho)/(h*uh)
    nut=.09*k.square()/eps.clamp_min(1e-9)
    gu=torch.stack([grad(u[:,i],q) for i in range(3)],1)
    divergence=gu[:,0,0]+gu[:,1,1]+gu[:,2,2]
    gp=grad(p+2*k/3,q)
    # Include the isotropic turbulent stress in modified pressure, since p is Fluent static pressure.
    strain=gu+gu.transpose(1,2)
    stress=(nu+nut)[:,None,None]*strain
    momentum=[]
    for i in range(3):
        divstress=sum(grad(stress[:,i,j],q)[:,j] for j in range(3))
        momentum.append((u*gu[:,i,:]).sum(-1)+gp[:,i]-divstress)
    production=.5*nut*(strain.square().sum((1,2)))
    gk=grad(k,q);ge=grad(eps,q)
    dk=sum(grad((nu+nut)*gk[:,j],q)[:,j] for j in range(3))
    de=sum(grad((nu+nut/1.3)*ge[:,j],q)[:,j] for j in range(3))
    rk=(u*gk).sum(-1)-dk-production+eps
    re=(u*ge).sum(-1)-de-1.44*eps/k.clamp_min(1e-8)*production+1.92*eps.square()/k.clamp_min(1e-8)
    return torch.stack((divergence,*momentum,rk,re),-1)
