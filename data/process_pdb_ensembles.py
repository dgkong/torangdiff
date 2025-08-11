import gc
import multiprocessing as mp
import random
import shutil
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import FastMMCIFParser, PPBuilder, calc_dihedral
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from tqdm import tqdm

BASE_INPUT_DIR = Path("data/pdb_ensemble_downloads")
OUTPUT_DIR = Path("data/datasets")
CUTOFF_DATE = datetime(2024, 1, 1) # for benchmark set
SEED = 37
NEEDED_KEYS = {
    "_entity_poly.entity_id",
    "_entity_poly.pdbx_seq_one_letter_code_can",
    "_pdbx_database_status.recvd_initial_deposition_date",
    "_struct_asym.entity_id",
    "_struct_asym.id",
    "_pdbx_poly_seq_scheme.asym_id",
    "_pdbx_poly_seq_scheme.pdb_strand_id",
    "_pdbx_poly_seq_scheme.pdb_seq_num",
    "_pdbx_poly_seq_scheme.pdb_ins_code",
    "_pdbx_poly_seq_scheme.seq_id",
}

def load_cif_dict(path):
    raw = MMCIF2Dict(str(path))
    slim = {k: raw[k] for k in NEEDED_KEYS if k in raw}
    del raw
    return slim

def get_residue_mapping(entity_id, mmcif_dict):    
    candidate_asym_ids = []
    asym_entity_ids = mmcif_dict['_struct_asym.entity_id']
    asym_ids = mmcif_dict['_struct_asym.id']
    for i, eid in enumerate(asym_entity_ids):
        if eid == entity_id:
            candidate_asym_ids.append(asym_ids[i])

    for asym_id in candidate_asym_ids:
        mapping = {}
        scheme_indices = [i for i, s_id in enumerate(mmcif_dict['_pdbx_poly_seq_scheme.asym_id']) if s_id == asym_id]
        chain_id = mmcif_dict['_pdbx_poly_seq_scheme.pdb_strand_id'][scheme_indices[0]]
        for i in scheme_indices:
            res_num_str = mmcif_dict['_pdbx_poly_seq_scheme.pdb_seq_num'][i]
            res_num = int(res_num_str)
            ins_code = mmcif_dict['_pdbx_poly_seq_scheme.pdb_ins_code'][i]
            ins_code = ' ' if ins_code in ('.', '?') else ins_code

            canonical_idx = int(mmcif_dict['_pdbx_poly_seq_scheme.seq_id'][i])
            mapping[canonical_idx] = (res_num, ins_code)
        if mapping:
            return mapping, chain_id
    return None, None

def calculate_torsions(chain):
    torsions = {}
    polypeptides = PPBuilder().build_peptides(chain)
    for polypeptide in polypeptides:
        phi_psi_list = polypeptide.get_phi_psi_list()
        for i, res in enumerate(polypeptide):
            res_id = (res.get_id()[1], res.get_id()[2])
            phi, psi = phi_psi_list[i]
            torsions[res_id] = [phi, psi, None]

        for i in range(len(polypeptide) - 1):
            curr_res, next_res = polypeptide[i], polypeptide[i+1]
            try:
                omega = calc_dihedral(curr_res['CA'].get_vector(), curr_res['C'].get_vector(),
                                      next_res['N'].get_vector(), next_res['CA'].get_vector())
                curr_res_id = (curr_res.get_id()[1], curr_res.get_id()[2])
                torsions[curr_res_id][2] = omega
            except KeyError:
                continue
    return torsions

def _process_ensemble(ensemble_path):
    try:
        parser = FastMMCIFParser(QUIET=True)

        all_cif_files = [p.name for p in ensemble_path.glob("*.cif")]
        cif_files = all_cif_files
        if len(all_cif_files) > 16:
            cif_files = random.sample(all_cif_files, 16)

        cif_cache = {}
        all_seqs = {}
        earliest_date = None

        for cf in cif_files:
            entity_id = Path(cf).stem.split("_")[1]
            cf_path = ensemble_path / cf
            cif_dict = load_cif_dict(cf_path)
            cif_cache[cf] = cif_dict

            eids = cif_dict["_entity_poly.entity_id"]
            seqs = cif_dict["_entity_poly.pdbx_seq_one_letter_code_can"]
            for i, seq in enumerate(seqs):
                if eids[i] == entity_id:
                    all_seqs[cf] = seq.replace("\n", "")
                    break
            date_str = cif_dict["_pdbx_database_status.recvd_initial_deposition_date"][0]
            deposition_date = datetime.strptime(date_str, "%Y-%m-%d")
            if earliest_date is None or deposition_date < earliest_date:
                earliest_date = deposition_date
        set_name = "benchmark" if earliest_date >= CUTOFF_DATE else "train_val"

        # representative sequence from longest common subsequence
        all_seq_list = list(all_seqs.values())
        repr_seq = all_seq_list[0]
        for seq in all_seq_list[1:]:
            matcher = SequenceMatcher(None, repr_seq, seq, autojunk=False)
            i, _, m_len = matcher.find_longest_match(0, len(repr_seq), 0, len(seq))
            repr_seq = repr_seq[i: i + m_len]
            if len(repr_seq) < 50:
                return (False, ensemble_path, f"representative sequence error\n - seq: {seq}\n - rep: {repr_seq}")

        out_dir = OUTPUT_DIR / set_name / ensemble_path.name
        out_dir.mkdir(parents=True, exist_ok=True)

        for cf in cif_files:
            cf_path = ensemble_path / cf
            pdb_id = Path(cf).stem
            entity_id = pdb_id.split("_")[1]
            mapping, chain_id = get_residue_mapping(entity_id, cif_cache[cf])
            if not mapping:
                return (False, ensemble_path, f"mapping failed for {pdb_id} in {ensemble_path.name}")

            model = parser.get_structure(pdb_id, str(cf_path))[0]
            curr_seq = all_seqs[cf]
            offset = curr_seq.find(repr_seq)
            if offset == -1:
                return (False, ensemble_path, f"representative seq not found for {pdb_id} in {ensemble_path.name}")

            torsions = calculate_torsions(model[chain_id])
            aligned_torsions = np.full((len(repr_seq), 3), np.nan, dtype=np.float32)
            for i in range(len(repr_seq)):
                res_id = mapping.get(i + offset + 1)
                if res_id in torsions:
                    phi, psi, omega = torsions.get(res_id)
                    if phi is not None:
                        aligned_torsions[i, 0] = phi
                    if psi is not None:
                        aligned_torsions[i, 1] = psi
                    if omega is not None:
                        aligned_torsions[i, 2] = omega

            torch.save(aligned_torsions, out_dir / f"{pdb_id}.pt")
            del model, torsions, aligned_torsions
        gc.collect()
            
        with open(out_dir / "sequence.txt", "w") as f:
            f.write(repr_seq)
        return (True, ensemble_path, None)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        import traceback
        return (False, ensemble_path, traceback.format_exc())
    
def init_worker(seed):
    random.seed(seed)

def process_ensembles():
    ensemble_dirs = [p for p in BASE_INPUT_DIR.iterdir() if p.is_dir()]
    unprocessed_dirs = []
    for dir in ensemble_dirs:
        cif_count = sum(1 for f in dir.iterdir() if f.is_file() and f.suffix == ".cif")
        train_val_target = OUTPUT_DIR / "train_val" / dir.name
        benchmark_target = OUTPUT_DIR / "benchmark" / dir.name
        if not train_val_target.is_dir() and not benchmark_target.is_dir():
            unprocessed_dirs.append(dir)
            continue
        pt_count = 0 # find partially processed ensembles
        if train_val_target.is_dir():
            pt_count = sum(1 for f in train_val_target.iterdir() if f.is_file() and f.suffix == ".pt")
        elif benchmark_target.is_dir():
            pt_count = sum(1 for f in benchmark_target.iterdir() if f.is_file() and f.suffix == ".pt")
        if (cif_count <= 16 and cif_count != pt_count) or (cif_count > 16 and pt_count != 16):
            target_dir = train_val_target if train_val_target.is_dir() else benchmark_target
            shutil.rmtree(target_dir)
            unprocessed_dirs.append(dir)
    print(f"{len(unprocessed_dirs)} ensembles to process")

    num_workers = mp.cpu_count() - 1
    print(f"Starting parallel processing with {num_workers} workers")
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(SEED,), maxtasksperchild=20) as pool:
        results_iter = pool.imap_unordered(_process_ensemble, unprocessed_dirs)
        pbar = tqdm(results_iter, total=len(unprocessed_dirs), desc="Processing Ensembles")
        for success, dir_path, error_msg in pbar:
            if not success:
                tqdm.write(f"FAILED: {dir_path.name}")
                tqdm.write(f"- Reason: {error_msg}")
    print("Processing complete")

if __name__ == "__main__":
    random.seed(SEED)
    process_ensembles()
