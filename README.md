# torsiondiff

Diffusion on backbone torsion angles to sample alternative conformations.

Goal:
Given a protein sequence and a reference structure, generate a diverse set of alternative conformations that capture the ensemble of states accessible to the protein. The project builds on top of highly accurate static predictors (e.g. AlphaFold) but extends them by modeling conformational variability rather than a single snapshot.

Training:
- [x] Download and process protein ensembles from PDB
- [x] Conditioning on a reference structure's torsion angles and the shared sequence ESM2 embeddings
- [x] DDPM forward process: add noise to torsion angles
- [x] UNet predict torsion noise over the residue sequence

- ~1s per step with batch size 64
- Early stopping at step 16500. Final val loss: 0.037349. Total wallclock time: 18103.7s
- Best val loss at step 11500: 0.0338.

Evaluation:
- [ ] Find the torsion angle distance and RMSD tolerance (median of min(nearest neighbor distances)/residue)
- [ ] Generate k torsion angle conformations and build their 3D structures as well
- [ ] Calculate fraction of generations within the tolerance (precision) and fraction of data within the samples (diversity)
- [ ] Check generated backbone validity, both Ramachandran score and clash score
- [ ] Recreate(?) Ramachandran plot

### Data:
- PDB chains with multiple states
  
Inspo:
- https://github.com/bjing2016/alphaflow
- https://github.com/gcorso/torsional-diffusion

Diffusion Implementation:
- https://huggingface.co/blog/annotated-diffusion
