"""Leakage-safe temporal source allocation (Appendix A1/A2)."""
import numpy as np

def build_window(target,li,rntl,eligible,held_out=None,radius=3):
    if target not in li:raise ValueError('Target-year Li is required')
    shape=li[target].shape;images=[];source=[];padding=[];offset=[]
    allowed=set(eligible)-({held_out} if held_out is not None else set())
    for k in range(-radius,radius+1):
        year=target+k;reference=k!=0 and year in allowed and year in rntl
        x=rntl.get(year) if reference else li.get(year)
        if x is None:images.append(np.zeros(shape,np.float32));source.append(0);padding.append(0)
        else:
            if x.shape!=shape:raise ValueError('Temporal inputs must share grid')
            # Caller must fill/track spatial NoData before passing model tensors.
            images.append(np.asarray(x,np.float32));source.append(int(reference));padding.append(1)
        offset.append(k)
    return dict(x=np.stack(images),yd=np.asarray(offset,np.float32),sm=np.asarray(source,np.float32),pm=np.asarray(padding,bool))

def fold_years(held_out):
    years={2000,2003,2004,2006,2010}
    if held_out not in years:raise ValueError('Unknown reference fold')
    return sorted(years-{held_out})
