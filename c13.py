"""C13 from actual PREPROCESSING-INPUT rasters; never reconstruct missing values.
Config: annual paths, independent study_mask, lower threshold, quantile population.
Keeps fixed temporal intersection; supports sequential (upper vs low-only) baseline.
Uses in-memory annual arrays; RAM approx ~3GB at Africa dimensions.
"""
import argparse,json
from pathlib import Path
from contextlib import ExitStack
import numpy as np,pandas as pd,rasterio
from metrics import valid,growth

def run(config):
    c=json.loads(Path(config).read_text());paths={int(k):v for k,v in c['annual'].items()};years=sorted(paths);rows=[]
    with ExitStack() as st:
        ds={y:st.enter_context(rasterio.open(paths[y])) for y in years};study=st.enter_context(rasterio.open(c['study_mask']));grid=ds[years[0]]
        for d in list(ds.values())+[study]:
            if (d.shape,d.crs,d.transform)!=(grid.shape,grid.crs,grid.transform):raise ValueError('Grid mismatch')
        z=study.read(1);m=valid(z,study.nodata)&(z>0)
        for d in ds.values():m &=valid(d.read(1),d.nodata)
        n=int(m.sum())
        if not n:raise ValueError('Empty fixed domain')
        for y,d in ds.items():
            x=d.read(1);base=x[m].astype(float)
            if np.any(base<0):raise ValueError('Negative radiance: apply declared shared baseline cleaning')
            basis=valid(x,d.nodata)&((x>1) if c.get('quantile_population')=='gt1' else (x>=c.get('default_lower',.1)))
            # Native annual quantile population is explicit; optional fixed-domain quantile.
            if c.get('quantile_domain','native')=='fixed':basis &=m
            cap=np.float32(np.quantile(x[basis].astype(float),c.get('upper_quantile',.9995),method='linear'))
            lo=base.copy();lo[lo<c.get('lower',.1)]=0;high=np.minimum(base,cap);both=np.minimum(lo,cap)
            for op,after,before in [('Lower-end thresholding',lo,base),('Upper-tail clipping',both if c.get('upper_baseline','lower_only')=='lower_only' else high,lo if c.get('upper_baseline','lower_only')=='lower_only' else base),('Combined preprocessing',both,base)]:
                b=float(before.sum());a=float(after.sum());rows.append(dict(year=y,operation=op,N_valid=n,N_affected=int(np.count_nonzero(after!=before)),affected_percent=100*np.mean(after!=before),TNLI_before=b,TNLI_after=a,TNLI_change_percent=100*(a-b)/b if b else np.nan,upper_cap=float(cap)))
    df=pd.DataFrame(rows)
    for op,g in df.groupby('operation'):
        ix=g.sort_values('year').index;gb=growth(df.loc[ix,'year'],df.loc[ix,'TNLI_before']);ga=growth(df.loc[ix,'year'],df.loc[ix,'TNLI_after']);df.loc[ix,'growth_before_percent']=gb;df.loc[ix,'growth_after_percent']=ga;df.loc[ix,'absolute_growth_change_pp']=abs(ga-gb)
    summary=df.groupby('operation',sort=False).agg(N_years=('year','size'),N_intervals=('absolute_growth_change_pp','count'),affected_mean=('affected_percent','mean'),affected_SD=('affected_percent','std'),TNLI_mean=('TNLI_change_percent','mean'),TNLI_SD=('TNLI_change_percent','std'),growth_mean=('absolute_growth_change_pp','mean'),growth_SD=('absolute_growth_change_pp','std')).reset_index()
    out=Path(c['output']);out.mkdir(parents=True,exist_ok=True);df.to_csv(out/'annual.csv',index=False);summary.to_csv(out/'summary.csv',index=False);(out/'configuration.json').write_text(json.dumps(c,indent=2));return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);print(run(p.parse_args().config).to_string(index=False))
