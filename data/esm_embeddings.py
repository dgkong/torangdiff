from pathlib import Path

import esm
import torch
from tqdm import tqdm

ROOTS = ["data/datasets/train_val", "data/datasets/benchmark"]
OUTDIR = Path("data/esm_embeddings")
OUTDIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "esm2_t33_650M_UR50D"  # 1280 embedding dim
LAYER = 33 # representation layer
BATCH_SIZE = 8
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

def iter_sequences():
    for root in ROOTS:
        for d in Path(root).iterdir():
            if not d.is_dir(): 
                continue
            seq_path = d / "sequence.txt"
            out = OUTDIR / f"{d.name}_esm2.pt"
            if out.exists():
                continue
            seq = seq_path.read_text()
            yield d.name, seq

def run_batch(model, batch_converter, names, seqs):
    batch = [(n, s) for n, s in zip(names, seqs)]
    _, _, toks = batch_converter(batch) # (B, L+2) with BOS/EOS
    toks = toks.to(DEVICE)
    with torch.inference_mode():
        out = model(toks, repr_layers=[LAYER], return_contacts=False)
        reps = out["representations"][LAYER] # (B, L+2, 1280)
        for i, (name, seq) in enumerate(zip(names, seqs)):
            L = len(seq)
            rep = reps[i, 1:L+1].detach().to("cpu").to(torch.float16)
            torch.save(rep, OUTDIR / f"{name}_esm.pt")

def main():
    print(f"Loading {MODEL_NAME}")
    model, alphabet = getattr(esm.pretrained, MODEL_NAME)()
    model = model.to(DEVICE).eval()
    batch_converter = alphabet.get_batch_converter()

    total = sum(1 for root in ROOTS for p in Path(root).iterdir() if p.is_dir())

    names, seqs = [], []
    with tqdm(total=total, desc="Embedding sequences", unit="seq") as pbar:
        for name, seq in iter_sequences():
            names.append(name)
            seqs.append(seq)
            if len(seqs) == BATCH_SIZE:
                run_batch(model, batch_converter, names, seqs)
                pbar.update(len(seqs))
                names, seqs = [], []
        if seqs:
            run_batch(model, batch_converter, names, seqs)
            pbar.update(len(seqs))

if __name__ == "__main__":
    main()
