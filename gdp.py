"""Country-wise level and first-difference OLS with common product-year samples."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd

def regress(panel,products):
    required=['ISO3','year','gdp','Subregion']+products
    if not set(required)<=set(panel):raise ValueError(f'Required columns: {required}')
    if panel.duplicated(['ISO3','year']).any():raise ValueError('Duplicate country-year')
    rows=[]
    for iso,g in panel.groupby('ISO3'):
        g=g.sort_values('year');m=np.isfinite(g[['gdp']+products]).all(axis=1)&(g.gdp>0)&(g[products]>=0).all(axis=1);g=g[m];y=np.log(g.gdp.to_numpy());adj=np.diff(g.year.to_numpy())==1
        for product in products:
            x=np.log1p(g[product].to_numpy())
            for typ,a,b in [('level',x,y),('first_difference',np.diff(x)[adj],np.diff(y)[adj])]:
                r2=slope=None
                if len(a)>=3 and np.std(a)>0 and np.std(b)>0:slope=float(np.polyfit(a,b,1)[0]);r2=float(np.corrcoef(a,b)[0,1]**2)
                rows.append(dict(ISO3=iso,Subregion=g.Subregion.iloc[0] if len(g) else None,product=product,regression=typ,N=len(a),R2=r2,slope=slope))
    d=pd.DataFrame(rows)
    # Same countries for every product and both analyses in the published summaries.
    keep=d.groupby('ISO3').R2.apply(lambda z:z.notna().all());d['included']=d.ISO3.map(keep)
    s=d[d.included].groupby(['Subregion','product','regression']).agg(countries=('ISO3','nunique'),R2=('R2','mean')).reset_index()
    all_=d[d.included].groupby(['product','regression']).agg(countries=('ISO3','nunique'),R2=('R2','mean')).reset_index();all_['Subregion']='Africa'
    return d,pd.concat([s,all_],ignore_index=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--panel',required=True);p.add_argument('--products',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();d,s=regress(pd.read_csv(a.panel),a.products);o=Path(a.output);o.mkdir(parents=True,exist_ok=True);d.to_csv(o/'countries.csv',index=False);s.to_csv(o/'summary.csv',index=False)
