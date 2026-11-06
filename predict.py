"""Inference on prepared tensors; uses explicit checkpoint architecture metadata."""
import argparse
import numpy as np
import torch
from models import StageI,StageII
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    c=torch.load(a.checkpoint,map_location='cpu',weights_only=True);stage=c['stage'];model=StageI(c['width']) if stage==1 else StageII(c['width'],c['population_stats']);model.load_state_dict(c['state_dict']);model.eval();d=np.load(a.input,allow_pickle=False);keys=['x','yd','sm','pm'] if stage==1 else ['stage1','population','geography','pop_valid']
    with torch.no_grad():y=model(*[torch.as_tensor(d[k],dtype=torch.float32) for k in keys]).numpy()
    # Keep pre-cutoff output: C13 requires it. Release cutoff is a separate operation.
    np.savez_compressed(a.output,prediction=y)
