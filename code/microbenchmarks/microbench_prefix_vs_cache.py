#!/usr/bin/env python3
import argparse, csv, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset

MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

class Indexed(Dataset):
    def __init__(self, ds): self.ds=ds
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        x,y=self.ds[i]; return x,y,i

class Split(nn.Module):
    def __init__(self, m): super().__init__(); self.m=m
    def prefix(self, x, b):
        m=self.m
        mods=[nn.Sequential(m.conv1,m.bn1,m.relu,m.maxpool)]
        if b in ["layer1","layer2","layer3","layer4"]: mods.append(m.layer1)
        if b in ["layer2","layer3","layer4"]: mods.append(m.layer2)
        if b in ["layer3","layer4"]: mods.append(m.layer3)
        if b=="layer4": mods.append(m.layer4)
        out=x
        for mod in mods: out=mod(out)
        return out

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def model(num_classes):
    try:
        from torchvision.models import ResNet50_Weights
        m=torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    except Exception:
        m=torchvision.models.resnet50(pretrained=True)
    m.fc=nn.Linear(m.fc.in_features,num_classes)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cifar10","cifar100"], required=True)
    ap.add_argument("--data_root", default="/mnt/data2/data")
    ap.add_argument("--boundary", choices=["layer3","layer4"], required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf=T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor(),T.Normalize(MEAN,STD)])
    cls,nc=(torchvision.datasets.CIFAR10,10) if args.dataset=="cifar10" else (torchvision.datasets.CIFAR100,100)
    ds=cls(root=args.data_root,train=True,download=False,transform=tf)
    dl=DataLoader(Indexed(ds),batch_size=args.batch_size,shuffle=True,num_workers=4,pin_memory=True)
    x,_,_=next(iter(dl)); x=x.to(device,non_blocking=True)
    sp=Split(model(nc).to(device).eval()).eval()

    with torch.no_grad():
        feat=sp.prefix(x,args.boundary); sync()
    shape=tuple(feat.shape[1:])
    bank_gb=len(ds)*np.prod(shape)*2/(1024**3)
    slots=max(args.batch_size*4,256)
    bank=torch.randn((slots,*shape),device=device,dtype=torch.float16)
    idx=torch.randint(0,slots,(args.batch_size,),device=device)

    with torch.no_grad():
        for _ in range(args.warmup):
            _=sp.prefix(x,args.boundary)
            _=bank.index_select(0,idx).clone()
        sync()
        pt=[]; lt=[]
        for _ in range(args.iters):
            t=time.perf_counter(); _=sp.prefix(x,args.boundary); sync(); pt.append(time.perf_counter()-t)
            t=time.perf_counter(); _=bank.index_select(0,idx).clone(); sync(); lt.append(time.perf_counter()-t)

    def stat(v):
        a=np.asarray(v)*1000
        return {"mean_ms":float(a.mean()),"median_ms":float(np.median(a)),"p90_ms":float(np.percentile(a,90))}
    res={"dataset":args.dataset,"boundary":args.boundary,"batch_size":args.batch_size,
         "feature_shape":shape,"estimated_full_bank_gb":float(bank_gb),
         "prefix_forward":stat(pt),"gpu_bank_lookup_clone":stat(lt),
         "speedup_lookup_vs_prefix":float((np.mean(pt))/(max(np.mean(lt),1e-12)))}
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output_json).write_text(json.dumps(res,indent=2))
    with open(args.output_csv,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=["dataset","boundary","batch_size","feature_shape","estimated_full_bank_gb","prefix_mean_ms","lookup_mean_ms","speedup_lookup_vs_prefix"])
        w.writeheader()
        w.writerow({"dataset":args.dataset,"boundary":args.boundary,"batch_size":args.batch_size,"feature_shape":str(shape),
                    "estimated_full_bank_gb":res["estimated_full_bank_gb"],
                    "prefix_mean_ms":res["prefix_forward"]["mean_ms"],
                    "lookup_mean_ms":res["gpu_bank_lookup_clone"]["mean_ms"],
                    "speedup_lookup_vs_prefix":res["speedup_lookup_vs_prefix"]})
    print(json.dumps(res,indent=2))
if __name__=="__main__": main()
