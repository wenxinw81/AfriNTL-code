"""Windowed fixed-domain comparisons, including BM accepted-area masks."""
import argparse,json
from contextlib import ExitStack
from pathlib import Path
import numpy as np,rasterio
from rasterio.windows import Window
from metrics import valid

def evaluate(reference,products,threshold=.1,coverage=None,min_coverage=.8,domain=None):
    with ExitStack() as st:
        ref=st.enter_context(rasterio.open(reference));ds={k:st.enter_context(rasterio.open(v)) for k,v in products.items()}
        cov=st.enter_context(rasterio.open(coverage)) if coverage else None
        dm=st.enter_context(rasterio.open(domain)) if domain else None
        for d in list(ds.values())+[x for x in [cov,dm] if x is not None]:
            if (d.shape,d.crs,d.transform)!=(ref.shape,ref.crs,ref.transform):raise ValueError('Grid mismatch; explicitly align upstream')
        sums={k:np.zeros(9) for k in ds};counts={k:np.zeros(4,dtype=np.int64) for k in ds};totals={k:0. for k in ds};rt=0.;nf=0
        for row in range(0,ref.height,128):
            w=Window(0,row,ref.width,min(128,ref.height-row));y=ref.read(1,window=w);ps={k:d.read(1,window=w) for k,d in ds.items()};m=valid(y,ref.nodata)
            for k,x in ps.items():m &=valid(x,ds[k].nodata)
            if cov is not None:
                c=cov.read(1,window=w);m &=valid(c,cov.nodata)&(c>=min_coverage)
            if dm is not None:
                x=dm.read(1,window=w);m &=valid(x,dm.nodata)&(x>0)
            if np.any(y[m]<0) or any(np.any(x[m]<0) for x in ps.values()):raise ValueError('Declare negative handling upstream')
            light=m&(y>=threshold);background=m&~light;yy=y[light].astype(float);z=np.log1p(yy);nf+=int(m.sum());rt+=y[m].sum(dtype=float)
            for k,x in ps.items():
                pp=x[light].astype(float);u=np.log1p(pp);e=pp-yy;sums[k]+=[len(pp),z.sum(),u.sum(),(z*z).sum(),(u*u).sum(),(z*u).sum(),(e*e).sum(),e.sum(),np.abs(e).sum()]
                on=x>=threshold;counts[k]+=[(light&on).sum(),(light&~on).sum(),(background&on).sum(),(background&~on).sum()];totals[k]+=x[m].sum(dtype=float)
        out={};ratio=lambda a,b:float(100*a/b) if b else None
        for k,s in sums.items():
            n,sx,sy,sxx,syy,sxy,se,b,ae=s;den=(sxx-sx*sx/n)*(syy-sy*sy/n) if n else 0;tp,fn,fp,tn=map(int,counts[k])
            out[k]=dict(N_light=int(n),N_full=nf,r2=(sxy-sx*sy/n)**2/den if den>0 else None,RMSE=float(np.sqrt(se/n)) if n else None,Bias=float(b/n) if n else None,MAE=float(ae/n) if n else None,TP=tp,FN=fn,FP=fp,TN=tn,OE=ratio(fn,tp+fn),CE=ratio(fp,tp+fp),FPR=ratio(fp,fp+tn),TNLI_reference=float(rt),TNLI_prediction=float(totals[k]),TNLI_relative_error=ratio(totals[k]-rt,rt))
        return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--product',action='append',required=True,help='name=path, repeat for common-domain comparisons');p.add_argument('--coverage');p.add_argument('--min-coverage',type=float,default=.8);p.add_argument('--domain');p.add_argument('--threshold',type=float,default=.1);p.add_argument('--output',required=True);a=p.parse_args();result=evaluate(a.reference,dict(v.split('=',1) for v in a.product),a.threshold,a.coverage,a.min_coverage,a.domain);Path(a.output).write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
