# PDB Ensembles

Used the PDB Search API to query protein ensembles.

### Query (8/10):
- length between 50 and 400 (typical single-domain globular proteins, anything smaller is likely a peptide and anything larger is likely a multi-domain assembly)
- resolution less than 2.5 angstrom ("high resolution")
- 10,000 total groups is the max for one search query
- returns 3185 ensembles (group count >= 2) consisting of 17,208 total structures

### Processing:
- randomly sampled 16 proteins for ensembles with more than 16 (151 of these ensembles)
- for the sake of exact residue to torsion angle alignment between proteins, set the representative sequence as the longest subsequence
    - filtered longest subsequences shorter than 50
- resulted in 3055 train_val ensembles and 113 benchmark ensembles consisting of 13,060 total structures
    - benchmark set was split based on initial deposition date: 2024-01-01
- filtered 17 ensembles, 3 of them due to MMCIF key error and the rest due to longest subsequences shorter than 50

