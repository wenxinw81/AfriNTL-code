"""Reference implementation of Appendix A; engineering widths are configurable.
This source is not a recovered historical training implementation/checkpoint.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Sequential):
    def __init__(self,a,b,k=3):super().__init__(nn.Conv2d(a,b,k,padding=k//2),nn.GELU(),nn.Conv2d(b,b,3,padding=1),nn.GELU())

class StageI(nn.Module):
    """Shared U-Net annual encoding + per-location temporal multihead attention.
    x B,T,1,H,W; yd/sm/pm B,T. Target-year query is temporal center.
    """
    def __init__(self,width=32,heads=4):
        super().__init__();assert width%heads==0
        self.enc=Conv(1,width);self.down=Conv(width,width*2);self.up=Conv(width*3,width)
        self.project=nn.Linear(width+2,width);self.attn=nn.MultiheadAttention(width,heads,batch_first=True);self.head=nn.Conv2d(width,1,1)
    def forward(self,x,yd,sm,pm):
        b,t,_,h,w=x.shape
        if not torch.all(pm[:,t//2].bool()):raise ValueError('Target temporal position must exist')
        if torch.any(sm[:,t//2]!=0):raise ValueError('Target-year input must be Li')
        f=self.enc(x.reshape(b*t,1,h,w));z=self.down(F.avg_pool2d(f,2));z=F.interpolate(z,size=(h,w),mode='bilinear',align_corners=False);f=self.up(torch.cat([f,z],1)).view(b,t,-1,h,w)
        # Flatten only spatial locations; temporal attention never mixes pixels.
        f=f.permute(0,3,4,1,2);desc=torch.stack([yd,sm],-1)[:,None,None].expand(b,h,w,t,2)
        f=self.project(torch.cat([f,desc],-1)).reshape(b*h*w,t,-1)
        padding=(~pm.bool())[:,None,None,:].expand(b,h,w,t).reshape(b*h*w,t)
        out,_=self.attn(f[:,t//2:t//2+1],f,f,key_padding_mask=padding,need_weights=False)
        out=out[:,0].reshape(b,h,w,-1).permute(0,3,1,2)
        return F.softplus(self.head(out))

class PopulationSupport(nn.Module):
    def __init__(self,width,stats):
        super().__init__();self.c3=nn.Conv2d(width,width,3,padding=1);self.c5=nn.Conv2d(width,width,5,padding=2);self.weight=nn.Conv2d(2,1,1)
        self.register_buffer('mean',torch.tensor(stats['mean']).view(1,2,1,1));self.register_buffer('std',torch.tensor(stats['std']).view(1,2,1,1))
    def descriptors(self,pop,valid):
        valid=valid.bool();v=valid.to(pop.dtype);p=torch.where(valid,pop,0);n=F.avg_pool2d(v,7,1,3)*49
        mean=(F.avg_pool2d(p,7,1,3)*49)/n.clamp_min(1)
        second=(F.avg_pool2d(p*p,7,1,3)*49)/n.clamp_min(1)
        sd=(second-mean.square()).clamp_min(0).sqrt();cv=torch.where(mean>=1,sd/mean.clamp_min(1),0)
        desc=torch.cat([torch.log1p(mean),torch.log1p(cv)],1)
        return desc,n>0
    def forward(self,f,pop,valid):
        desc,context_valid=self.descriptors(pop,valid);z=(desc-self.mean)/self.std.clamp_min(1e-8);omega=torch.sigmoid(self.weight(z))
        return (1-omega)*self.c3(f)+omega*self.c5(f),context_valid

class Enhance(nn.Module):
    def __init__(self,c):
        super().__init__();self.res=Conv(c,c);self.spatial=nn.Conv2d(2,1,7,padding=3);self.channel=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(c,max(1,c//4),1),nn.GELU(),nn.Conv2d(max(1,c//4),c,1),nn.Sigmoid())
    def forward(self,f):
        u=f+self.res(f);gate=torch.sigmoid(self.spatial(torch.cat([u.mean(1,keepdim=True),u.amax(1,keepdim=True)],1)));u=u*gate
        return u*self.channel(u)

class Denoise(nn.Module):
    """7x7 feature similarity and learned intensity response. Chunk inputs to limit memory."""
    def __init__(self,c):
        super().__init__();self.phi=nn.Conv2d(c,c,1);self.rho=nn.Conv2d(c,1,1);self.value=nn.Conv2d(c,c,1);self.out=nn.Conv2d(c,c,1)
        initial=math.log(math.expm1(1.0));self.ts=nn.Parameter(torch.tensor(initial));self.ti=nn.Parameter(torch.tensor(initial))
    def forward(self,f,valid):
        b,c,h,w=f.shape;phi=self.phi(f);rho=self.rho(f)
        unfold=lambda x:F.unfold(x,7,padding=3).reshape(b,x.shape[1],49,h,w)
        neighbors=unfold(phi);sim=-(neighbors-phi[:,:,None]).square().sum(1)/F.softplus(self.ts).clamp_min(1e-6)
        intensity=-(unfold(rho)-rho[:,:,None]).abs().squeeze(1)/F.softplus(self.ti).clamp_min(1e-6)
        ok=unfold(valid.to(f.dtype)).squeeze(1)>0;score=(sim+intensity).masked_fill(~ok,-torch.inf)
        # Entirely invalid neighborhoods receive zero context, never NaN softmax.
        has=ok.any(1,keepdim=True);score=torch.where(has,score,torch.zeros_like(score));weights=torch.softmax(score,1)*ok
        context=(unfold(self.value(f))*weights[:,None]).sum(2)
        return f+self.out(context)

class StageII(nn.Module):
    def __init__(self,width=32,population_stats=None):
        super().__init__()
        if population_stats is None:raise ValueError('Supply training-only population descriptor mean/std')
        self.encode=Conv(1,width);self.support=PopulationSupport(width,population_stats);self.enhance=Enhance(width);self.denoise=Denoise(width);self.geo=Conv(2,width)
        self.fusion=nn.Conv2d(2*width,1,1);self.head=nn.Conv2d(width,1,1)
    def forward(self,stage1,population,geography,pop_valid):
        ntl=F.interpolate(stage1,size=population.shape[-2:],mode='bilinear',align_corners=False);f=self.encode(ntl);f,valid=self.support(f,population,pop_valid)
        geo=self.geo(geography);c=torch.sigmoid(self.fusion(torch.cat([f,geo],1)));e=self.enhance(f);d=self.denoise(f,valid)
        return F.softplus(self.head(c*e+(1-c)*d))
