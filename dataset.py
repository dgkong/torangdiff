import random
from pathlib import Path

import torch
from einops import rearrange, repeat
from torch.utils.data import Dataset

TRAIN_DIR = Path("data/datasets/train")
VAL_DIR = Path("data/datasets/val")
BENCHMARK_DIR = Path("data/datasets/benchmark")
ESM_DIR      = Path("data/esm_embeddings")

def process_angles(ang):
    valid = ~torch.isnan(ang) # (L,3)
    ang = torch.nan_to_num(ang, nan=0.0)
    ang_cs = rearrange([torch.cos(ang), torch.sin(ang)], "cs L d -> (d cs) L")
    valid_mask = repeat(valid, 'L d -> (d dup) L', dup=2)
    return ang_cs, valid_mask # (6,L)


class TorsionPairDataset(Dataset):
    def __init__(self, split):
        assert split in {'train', 'val'}
        self.split = split
        if split == 'train':
            self.root = TRAIN_DIR
        else:
            self.root = VAL_DIR
        self.ensembles = []
        for d in sorted(self.root.iterdir()):
            if not d.is_dir():
                continue
            conformers = sorted([p.name for p in d.glob("*.pt") if p.name != "sequence.txt"])
            self.ensembles.append({"dir": d, "conformers": conformers})

    def __len__(self):
        return len(self.ensembles)
    
    def _sample_train(self):
        ens = self.ensembles[random.randrange(len(self.ensembles))]
        d, conformers = ens["dir"], ens["conformers"]
        ref, target = random.sample(conformers, 2)
        return ref, target, d
    
    def _sample_val(self, index):
        ens = self.ensembles[index]
        d, conformers = ens["dir"], ens["conformers"]
        ref, target = conformers[0], conformers[-1]
        return ref, target, d

    def __getitem__(self, index):
        if self.split == 'train':
            ref, target, d = self._sample_train()
        else:
            ref, target, d = self._sample_val(index)

        ref_ang = torch.load(d / ref) # (L,3)
        target_ang = torch.load(d / target) # (L,3)
        xref, mask_ref = process_angles(ref_ang) # (6,L)
        x0, mask_0 = process_angles(target_ang) # (6,L)
        valid_mask = mask_ref & mask_0

        esm_emb = torch.load(ESM_DIR / f"{d.name}_esm.pt") # (L,d_emb), float16
        return {
            "x0": x0,
            "xref": xref,
            "mask": valid_mask,
            "esm": esm_emb,
            "ensemble": d.name,
            "ref": ref,
            "target": target,
        }
