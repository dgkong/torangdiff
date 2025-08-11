# torsiondiff

Diffusion on backbone torsion angles to sample alternative conformations. 

- [ ] Extract and add noise to the residue angles (phi, psi, omega)
- [ ] Conditioning on ESM2 embeddings for each residue token
- [ ] UNet over the residue sequence that denoises noisy torsions

### Data:
- PDB chains with multiple states
  
Inspo:
- https://github.com/gcorso/torsional-diffusion
- https://github.com/bjing2016/alphaflow
