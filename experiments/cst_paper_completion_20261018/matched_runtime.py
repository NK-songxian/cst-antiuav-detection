"""Matched CST input ablation: same lossless PNGs, labels, sampler, augmentation and model."""
from pathlib import Path
from copy import copy
import numpy as np
import torch
from ultralytics.data.dataset import YOLODataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import colorstr
from ultralytics.utils.torch_utils import unwrap_model
import sys
ROOT=Path('/root/autodl-tmp/Ryan/cst_antiuav')
sys.path.insert(0,str(ROOT/'experiments/aat_temporal_100e_20261017'))
import train_innovation as ti
INPUT_MODE='temporal'

def current_only_bgr(image):
    """Disk PNG BGR order is [far,near,current]; select before augmentation."""
    assert image.ndim==3 and image.shape[2]==3
    return np.ascontiguousarray(np.repeat(image[:,:,2:3],3,axis=2))

class MatchedDataset(YOLODataset):
    def __init__(self,*args,input_mode='temporal',**kwargs):
        self.input_mode=input_mode
        super().__init__(*args,**kwargs)
    def load_image(self,i,rect_mode=True):
        image,hw0,hw=super().load_image(i,rect_mode)
        if self.input_mode=='current':image=current_only_bgr(image)
        return image,hw0,hw

def make_dataset(args,img_path,batch,data,mode,stride=32,input_mode='temporal'):
    return MatchedDataset(img_path=img_path,imgsz=args.imgsz,batch_size=batch,
        augment=mode=='train',hyp=args,rect=args.rect or mode=='val',cache=args.cache or None,
        single_cls=args.single_cls or False,stride=stride,pad=0. if mode=='train' else .5,
        prefix=colorstr(f'{mode}: '),task=args.task,classes=args.classes,data=data,
        fraction=1.,input_mode=input_mode)

class MatchedValidator(DetectionValidator):
    def preprocess(self,batch):
        batch=super().preprocess(batch)
        # Also applies to trainer.final_eval() when it rebuilds a standalone loader.
        if INPUT_MODE=='current':batch['img'][:,1:]=batch['img'][:,:1].clone()
        return batch

class MatchedTrainer(ti.InnovationTrainer):
    def build_dataset(self,img_path,mode='train',batch=None):
        return make_dataset(self.args,img_path,batch,self.data,mode,
            stride=max(int(unwrap_model(self.model).stride.max()),32),input_mode=INPUT_MODE)
    def get_dataloader(self,dataset_path,batch_size=16,rank=-1,mode='train'):
        old=self.args.workers
        try:
            if mode!='train':
                batch_size=min(batch_size,128);self.args.workers=2
            return super().get_dataloader(dataset_path,batch_size,rank,mode)
        finally:self.args.workers=old
    def get_validator(self):
        args=copy(self.args);args.workers=2
        return MatchedValidator(self.test_loader,save_dir=self.save_dir,args=args,_callbacks=self.callbacks)
