import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.data import DataLoader

from dataset import TorsionPairDataset
from diffusion import DDPM
from modules import ESMProj, UNet1D

CHECKPOINT_PATH = Path("out/checkpoint")
LOG_PATH = Path("out/log")

def get_args():
    parser = argparse.ArgumentParser(description="Train a protein torsion angle diffusion model")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, help="Path to resume checkpoint")
    return parser.parse_args()

def save_checkpoint(model, esm_proj, optimizer, iteration, out):
    print("Saving checkpoint...")
    checkpoint = {
        "model": model.state_dict(),
        "esm_proj": esm_proj.state_dict(),
        "optim": optimizer.state_dict(),
        "iter": iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(src, model, esm_proj, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    esm_proj.load_state_dict(checkpoint["esm_proj"])
    optimizer.load_state_dict(checkpoint["optim"])
    return checkpoint["iter"]

def collate_pad(batch):
    max_len = max(b["x0"].shape[1] for b in batch)
    B = len(batch)
    x0 = torch.zeros(B, 6, max_len)
    xref = torch.zeros(B, 6, max_len)
    valid_mask = torch.zeros(B, 6, max_len, dtype=torch.bool)
    esm = torch.zeros(B, max_len, 1280)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        L = b["x0"].shape[1]
        x0[i,:,:L] = b["x0"]
        xref[i,:,:L] = b["xref"]
        valid_mask[i,:,:L] = b["mask"]
        esm[i,:L,:] = b["esm"]
        pad_mask[i,:L] = True

    return x0, xref, valid_mask, esm, pad_mask

def infinite_iter(loader):
    while True:
        for batch in loader:
            yield batch

def masked_mse_loss(input, target, valid_mask):
    return ((input - target)**2 * valid_mask).sum() / valid_mask.sum().clamp(min=1)

def main():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

    # Load Config
    args = get_args()
    with open(args.config) as f:
        config = json.load(f)

    MAX_STEPS = config["max_steps"]
    VAL_INTERVAL = config["val_interval"]
    B = config["batch_size"]
    MIN_DELTA = config["min_relative_delta"]
    PATIENCE = config["patience"]

    # Load Data
    train_dataset = TorsionPairDataset('train')
    val_dataset = TorsionPairDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=B, collate_fn=collate_pad)
    val_loader = DataLoader(val_dataset, batch_size=B, collate_fn=collate_pad)

    # Load Model and Optimizer
    esm_proj = ESMProj(**config["esm_proj"]).to(device)
    model = UNet1D(**config["model"]).to(device)
    trainable_params = list(model.parameters())+list(esm_proj.parameters())
    optimizer = torch.optim.AdamW(trainable_params, **config["optimizer"])
    ddpm = DDPM(T=config["ddpm_T"], device=device)

    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    log_file = os.path.join(LOG_PATH, f"log.txt")
    if args.checkpoint is None:
        global_optimizer_step = 0
        with open(log_file, "w") as f:
            pass # open for writing to clear the file
    else:
        global_optimizer_step = load_checkpoint(args.checkpoint, model, esm_proj, optimizer)
    train_batches = infinite_iter(train_loader)
    best_val_loss = float('inf')
    patience_counter = 0

    def build_condition(esm, xref, pad_mask):
        esm = rearrange(esm_proj(esm), "b l c -> b c l")
        cond = torch.cat([xref, esm], dim=1) # (B,6+128,L)
        cond = cond * pad_mask.to(cond.dtype)
        return cond

    def run_validation():
        print("Running validation...")
        model.eval(); esm_proj.eval()
        val_loss_accum = 0.0
        val_steps = 0
        with torch.no_grad():
            for x0, xref, valid_mask, esm, pad_mask in val_loader:
                x0, xref, esm = x0.to(device), xref.to(device), esm.to(device)
                valid_mask, pad_mask = valid_mask.to(device), pad_mask.to(device)
                pad_mask = rearrange(pad_mask, "b l -> b 1 l")
                cond = build_condition(esm, xref, pad_mask)
                t = torch.randint(0, ddpm.T, (x0.shape[0],), device=device)
                x_t, eps = ddpm.add_noise(x0, t)
                eps_hat = model(x_t, cond, t)
                mask = (valid_mask & pad_mask).to(eps.dtype)
                loss = masked_mse_loss(eps_hat, eps, mask)
                val_loss_accum += loss.item()
                val_steps += 1
        model.train(); esm_proj.train()
        avg_loss = val_loss_accum / val_steps
        return avg_loss
    
    print(f"Starting training for {MAX_STEPS} optimizer steps...")
    start_time = time.time()
    while global_optimizer_step < MAX_STEPS:
        # VALIDATION AND EARLY StOPPING
        if global_optimizer_step > 0  and global_optimizer_step % VAL_INTERVAL == 0:
            val_loss = run_validation()
            print(f"step {global_optimizer_step:4d}/{MAX_STEPS} | val loss: {val_loss:.6f}")
            with open(log_file, "a") as f:
                f.write(f"{global_optimizer_step} val {val_loss:.6f}\n")
            if val_loss < best_val_loss * (1-MIN_DELTA):
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model.")
                best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(model, esm_proj, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, f"best_model.pt"))
            else:
                patience_counter += 1
                print(f"No significant improvement in validation loss. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter == PATIENCE:
                print(f"Early stopping triggered after {patience_counter} validation cycles with no improvement.")
                break

        # TRAINING
        t0 = time.time()
        optimizer.zero_grad()

        x0, xref, valid_mask, esm, pad_mask = next(train_batches)
        x0, xref, esm = x0.to(device), xref.to(device), esm.to(device)
        valid_mask, pad_mask = valid_mask.to(device), pad_mask.to(device)
        pad_mask = rearrange(pad_mask, "b l -> b 1 l")
        cond = build_condition(esm, xref, pad_mask)

        t = torch.randint(0, ddpm.T, (x0.shape[0],), device=device)
        x_t, eps = ddpm.add_noise(x0, t)
        eps_hat = model(x_t, cond, t)
        mask = (valid_mask & pad_mask).to(eps.dtype)
        loss = masked_mse_loss(eps_hat, eps, mask)
        loss.backward()
        norm = nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        t1 = time.time()
        dt = t1 - t0
        print(f"step {global_optimizer_step:4d}/{MAX_STEPS} | loss: {loss:.6f} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms")
        with open(log_file, "a") as f:
            f.write(f"{global_optimizer_step} train {loss:.6f}\n")
        global_optimizer_step += 1

    final_val_loss = run_validation()
    end_time = time.time()
    with open(log_file, "a") as f:
        f.write(f"{global_optimizer_step} val {final_val_loss:.6f}\nTotal wallclock time: {end_time - start_time:.1f}s")
    print(f"Trained finished. Final val loss: {final_val_loss}. Total wallclock time: {end_time - start_time:.1f}s")
    save_checkpoint(model, esm_proj, optimizer, global_optimizer_step, os.path.join(CHECKPOINT_PATH, 'final_model.pt'))

if __name__ == "__main__":
    main()

