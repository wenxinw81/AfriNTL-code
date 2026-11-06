import unittest,tempfile,json
from pathlib import Path
import numpy as np
from metrics import common_mask,radiometric,detection,growth
from temporal import build_window,fold_years
from preprocess import cutoff
class ProtocolTests(unittest.TestCase):
    def test_zero_predictions_remain(self):
        y=np.array([0,.2,2.]);p=np.array([1.,0,2.]);m=common_mask([y,p]);r=radiometric(y,p,m);self.assertEqual(r['N'],2);self.assertAlmostEqual(r['RMSE'],np.sqrt(.02));c=detection(y,p,m);self.assertEqual((c['TP'],c['FN'],c['FP'],c['TN']),(1,1,1,0))
    def test_no_prediction_filter(self):
        self.assertEqual(radiometric(np.array([1.]),np.array([0.]),np.array([True]))['N'],1)
    def test_no_gap_growth(self):
        v=growth([2000,2001,2003],[100,110,121]);self.assertAlmostEqual(v[1],10);self.assertTrue(np.isnan(v[2]))
    def test_reference_leakage(self):
        li={t:np.ones((1,4,4),np.float32) for t in range(1997,2007)};r={2000:np.full((1,4,4),999),2003:np.full((1,4,4),3)}
        w=build_window(2000,li,r,fold_years(2000),held_out=2000);self.assertEqual(w['sm'][3],0);self.assertEqual(w['sm'][6],1);self.assertEqual(w['x'][3,0,0,0],1)
    def test_cutoff_preserves_valid_zero(self):
        x=np.array([0,.05,.1,1,10],np.float32);v=np.ones(5,bool);y,c=cutoff(x,v,fixed_upper=5);self.assertTrue(np.array_equal(y,np.array([0,0,.1,1,5],np.float32)))
    def test_c13_sequential_baseline_and_gap(self):
        import rasterio
        from rasterio.transform import from_origin
        from c13 import run
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);profile=dict(driver='GTiff',height=2,width=2,count=1,dtype='float32',crs='EPSG:4326',transform=from_origin(0,2,1,1),nodata=-9999)
            for name,z in [('study',np.ones((2,2))),('2000',np.array([[0,.05],[1,10]])),('2001',np.array([[0,.06],[2,20]])),('2003',np.array([[0,.07],[3,30]]))]:
                with rasterio.open(p/f'{name}.tif','w',**profile) as f:f.write(z.astype('float32'),1)
            c=dict(annual={y:str(p/f'{y}.tif') for y in ['2000','2001','2003']},study_mask=str(p/'study.tif'),lower=.1,upper_quantile=.5,upper_baseline='lower_only',output=str(p/'results'));(p/'config.json').write_text(json.dumps(c));s=run(p/'config.json');self.assertTrue(s.N_intervals.eq(1).all())
class ModelTests(unittest.TestCase):
    def test_models_forward_backward_and_padding(self):
        import torch
        from models import StageI,StageII
        torch.set_num_threads(1);torch.manual_seed(42)
        s=StageI(8,2);x=torch.rand(1,7,1,8,8);yd=torch.arange(-3,4).float()[None];sm=torch.zeros(1,7);pm=torch.ones(1,7);pm[:,0]=0
        y=s(x,yd,sm,pm);other=x.clone();other[:,0]=1e5;self.assertTrue(torch.allclose(y,s(other,yd,sm,pm),atol=1e-5));self.assertEqual(tuple(y.shape),(1,1,8,8));y.mean().backward()
        m=StageII(8,{'mean':[0,0],'std':[1,1]});pop=torch.rand(1,1,16,16)*10;geo=torch.rand(1,2,16,16);mask=torch.ones_like(pop,dtype=torch.bool)
        z=m(y.detach(),pop,geo,mask);self.assertTrue(torch.isfinite(z).all());self.assertTrue((z>=0).all());z.mean().backward()
        z=m(y.detach(),pop,geo,torch.zeros_like(mask));self.assertTrue(torch.isfinite(z).all())
if __name__=='__main__':unittest.main()
