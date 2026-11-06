"""Manuscript Sections 3.4 and Appendices B/D: explicitly separated domains."""
import numpy as np

def valid(x, nodata=None):
    x = np.asarray(x)
    return np.isfinite(x) & ((x != nodata) if nodata is not None else True)

def common_mask(arrays, nodata=None):
    if not arrays: raise ValueError('At least one raster required')
    if any(x.shape != arrays[0].shape for x in arrays): raise ValueError('Shape mismatch')
    return np.logical_and.reduce([valid(x, n) for x,n in zip(arrays, nodata or [None]*len(arrays))])

def radiometric(reference, prediction, mask, threshold=0.1, log=True):
    """Predicted zeros retained. Nonnegative radiance is required on selected pixels."""
    m = mask & (reference >= threshold)
    y, p = reference[m].astype('float64'), prediction[m].astype('float64')
    if np.any(y < 0) or np.any(p < 0): raise ValueError('Negative selected radiance; declare cleaning before evaluation')
    if not len(y): return dict(N=0, r2=None, RMSE=None, Bias=None, MAE=None)
    z,w = (np.log1p(y),np.log1p(p)) if log else (y,p)
    r2 = float(np.corrcoef(z,w)[0,1]**2) if len(y)>1 and np.std(z)>0 and np.std(w)>0 else None
    error = p-y
    return dict(N=len(y),r2=r2,RMSE=float(np.sqrt(np.mean(error**2))),Bias=float(error.mean()),MAE=float(np.abs(error).mean()))

def detection(reference, prediction, mask, threshold=.1, native_dn=False):
    r = reference >= threshold
    p = prediction > 0 if native_dn else prediction >= threshold
    tp,fn,fp,tn = [int(np.count_nonzero(m & mask)) for m in [r&p,r&~p,~r&p,~r&~p]]
    ratio=lambda a,b:100*a/b if b else None
    return dict(TP=tp,FN=fn,FP=fp,TN=tn,OE=ratio(fn,tp+fn),CE=ratio(fp,tp+fp),FPR=ratio(fp,fp+tn))

def tnli(reference, prediction, mask):
    r=float(reference[mask].sum(dtype='float64'));p=float(prediction[mask].sum(dtype='float64'))
    return dict(N_full=int(mask.sum()),reference_total=r,prediction_total=p,signed_difference=p-r,absolute_difference=abs(p-r),relative_error_percent=100*(p-r)/r if r else None)

def growth(years, totals):
    years=np.asarray(years);totals=np.asarray(totals,dtype=float)
    out=np.full(len(totals),np.nan)
    for i in range(1,len(totals)):
        if years[i]-years[i-1]==1 and totals[i-1]>0:out[i]=100*(totals[i]/totals[i-1]-1)
    return out

def urban(reference, prediction, mask):
    c=detection(reference.astype(float),prediction.astype(float),mask,.5);tp,fn,fp,tn=[c[k] for k in ['TP','FN','FP','TN']];n=tp+fn+fp+tn
    if not n:return dict(OA=None,Kappa=None,IoU=None,Ra=None)
    oa=(tp+tn)/n;pe=((tp+fp)*(tp+fn)+(tn+fn)*(tn+fp))/n**2
    return dict(OA=oa,Kappa=(oa-pe)/(1-pe) if pe<1 else None,IoU=tp/(tp+fn+fp) if tp+fn+fp else None,Ra=1-abs((tp+fp)-(tp+fn))/(tp+fn) if tp+fn else None)
