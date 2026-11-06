"""Explicit lower thresholds and upper quantiles; valid zeros never become NoData."""
import numpy as np

def cutoff(values,valid,lower=.1,upper_quantile=.9995,population='positive_after_default',fixed_upper=None):
    if np.any(values[valid]<0):raise ValueError('Resolve negative-radiance policy before this operation')
    basis=valid & ((values>1) if population=='gt1' else (values>=.1))
    if fixed_upper is None:
        if not basis.any():raise ValueError('Empty upper-quantile population')
        cap=float(np.quantile(values[basis].astype(float),upper_quantile,method='linear'))
    else:cap=float(fixed_upper)
    result=values.copy();result[valid&(result<lower)]=0;result[valid&(result>cap)]=np.float32(cap)
    return result,cap
