"""Train pre-extracted NPZ patches with spatially separate train/validation sets.
NPZ keys: x,yd,sm,pm for Stage I; stage1,population,geography,pop_valid for II;
plus y,mask,block_id. Export/prepare patches independently using documented blocks.
"""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from models import StageI,StageII

class Patches(Dataset):
    def __init__(self,path,stage):
        self.data=np.load(path,allow_pickle=False);self.stage=stage
        self.inputs=['x','yd','sm','pm'] if stage==1 else ['stage1','population','geography','pop_valid']
        for key in self.inputs+['y','mask','block_id']:
            if key not in self.data:raise ValueError(f'Missing {key}')
    def __len__(self):return len(self.data['y'])
    def __getitem__(self,i):
        return {k:torch.as_tensor(self.data[k][i],dtype=torch.float32) for k in self.inputs+['y','mask']}

def loss_sum(pred,y,mask):
    m=mask.bool()&torch.isfinite(y)
    if not m.any():return None,0
    return (pred[m]-y[m]).abs().sum(),int(m.sum())

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[1,2],required=True);p.add_argument('--train',required=True);p.add_argument('--validation',required=True);p.add_argument('--output',required=True);p.add_argument('--population-stats');p.add_argument('--epochs',type=int,default=200);p.add_argument('--width',type=int,default=32);p.add_argument('--device',default='cpu');a=p.parse_args()
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    tr=Patches(a.train,a.stage);va=Patches(a.validation,a.stage)
    if set(tr.data['block_id'].tolist())&set(va.data['block_id'].tolist()):raise ValueError('Training/validation spatial blocks overlap')
    stats=json.loads(Path(a.population_stats).read_text()) if a.population_stats else None
    model=(StageI(a.width) if a.stage==1 else StageII(a.width,stats)).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4);scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs,eta_min=1e-6)
    train=DataLoader(tr,batch_size=32 if a.stage==1 else 16,shuffle=True);val=DataLoader(va,batch_size=16);best=float('inf');wait=0;history=[]
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        for batch in train:
            batch={k:v.to(a.device) for k,v in batch.items()};pred=model(*[batch[k] for k in tr.inputs]);total,n=loss_sum(pred,batch['y'],batch['mask'])
            if not n:continue
            opt.zero_grad();(total/n).backward();opt.step()
        model.eval();error=0.;nval=0
        with torch.no_grad():
            for batch in val:
                batch={k:v.to(a.device) for k,v in batch.items()};pred=model(*[batch[k] for k in va.inputs]);total,n=loss_sum(pred,batch['y'],batch['mask'])
                if n:error+=float(total);nval+=n
        if not nval:raise ValueError('No validation pixels')
        score=error/nval;history.append(dict(epoch=epoch+1,validation_L1=score));scheduler.step()
        if score<best:
            best=score;wait=0;torch.save(dict(state_dict=model.state_dict(),stage=a.stage,width=a.width,population_stats=stats,epoch=epoch+1,validation_L1=score,implementation='manuscript-reference-20261106'),out)
        else:wait+=1
        print(history[-1],flush=True)
        if wait>=20:break
    out.with_suffix('.history.json').write_text(json.dumps(history,indent=2))
if __name__=='__main__':main()
