#!/usr/bin/env python3
"""
batch_main.py
chews through gene2uniprot.csv, grabs the matching AlphaFold PDB from
./alphafold_pdbs/<UID>.pdb, and runs the full insertion-site analysis
for every UniProt ID.

dependencies: msa.py, features.py, annotations.py, scoring.py, SignalP6,
Stride, tqdm, matplotlib, scipy, biopython, requests.

layout:
  gene2uniprot.csv      (gene_symbol,uniprot_id)
  alphafold_pdbs/UID.pdb
"""

import csv
import sys
import ast
import subprocess
import pathlib
import time
import warnings

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
from tqdm import tqdm

import urllib3 ; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import msa
import features
import annotations
import scoring

# ------------ hard-wired paths ------------
CSV_MAP = sys.argv[1] if len(sys.argv) > 1 else "gene2uniprot.csv"
PDB_DIR    = pathlib.Path("alphafold_pdbs")
STRIDE_EXE = pathlib.Path("stride/stride")      # compiled stride binary
CHAIN_ID   = "A"

# ------------ helpers ------------
def fetch_record(uid: str) -> SeqRecord:
    """download FASTA from UniProt."""
    return msa.fetch_sequence_from_uniprot(uid)

def run_signalp(fasta_path: str):
    cmd = [
        "python3",
        "signalp6_fast/signalp-6-package/signalp/predict.py",
        "--fastafile", fasta_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return ast.literal_eval(res.stdout.strip())

def strip_sp(pdb_in: str, cut_idx: int) -> str:
    pdb_out = pdb_in.replace(".pdb", "_mature.pdb")
    with open(pdb_in) as fin, open(pdb_out, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                if int(line[22:26]) <= cut_idx:
                    continue
            fout.write(line)
    return pdb_out

# ------------ per-protein pipeline ------------
def analyse(uid: str, pdb_path: pathlib.Path):
    print(f"\n=== analysing {uid} ===")

    # 1. sequence + SP trimming ----------------------------------
    rec         = fetch_record(uid)
    full_seq    = str(rec.seq)
    fasta_tmp   = f"{uid}.fasta"
    SeqIO.write(rec, fasta_tmp, "fasta")

    cut = run_signalp(fasta_tmp)
    sp_present = cut[0] != -1

    pdb_file = str(pdb_path)
    if sp_present:
        pdb_file = strip_sp(pdb_file, cut[0])

    # 2. multiple alignment --------------------------------------
    msa_res, _ = msa.perform_msa(rec)
    msa_seqs = [str(s) for s in msa_res]
    if not msa_seqs:
        print("no MSA hits → skip")
        return
    query_aln = msa_seqs[0]

    # 3. per-residue features ------------------------------------
    entropy   = features.calculate_shannon_entropy(msa_seqs)
    extend    = features.calculate_extendable_scores(msa_seqs)
    disorder  = features.predict_disordered_binding_regions(full_seq)

    ss_sasa = features.get_secondary_structure_and_sasa(
        pdb_file, stride_executable=str(STRIDE_EXE), chain_id=CHAIN_ID
    )
    ss_sasa_list = features.parse_ss_sasa_for_chain(ss_sasa, chain_id=CHAIN_ID)

    feats = features.combine_features(
        query_aln, cut, entropy, extend, disorder, ss_sasa_list
    )

    # 4. annotations --------------------------------------------
    tm_rng, sp_rng = annotations.get_tm_sp(full_seq)
    ptm_pos        = annotations.get_ptm_positions(full_seq)

    # 5. scoring -------------------------------------------------
    length   = len(full_seq)
    ptm_pen  = scoring.assign_ptm_penalties(length, ptm_pos, base_penalty=1.0, distance_decay=True)
    excl_sp  = scoring.mark_excluded_positions(length, sp_rng)
    excl_tm  = scoring.mark_excluded_positions(length, tm_rng)

    ranked = scoring.combine_all_scores(
        full_seq, feats, ptm_pen, excl_sp.union(excl_tm), excl_sp, cut
    )

    # 6. outputs -------------------------------------------------
    prefix = uid
    csv_out = f"scores/{prefix}_scores.csv"
    with open(csv_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "Position", "Residue", "TotalPenalty",
            "PTMPenalty", "ConPen", "DisorderPen",
            "SSPenalty", "SASAPenalty", "Entropy",
            "Extension", "Disorder", "SecondaryStructure", "SASA",
        ])
        for r in ranked:
            w.writerow([
                r["position"], r["residue"], f"{r['total_penalty']:.2f}",
                f"{r['ptm_penalty']:.2f}", f"{r['conservation_penalty']:.2f}",
                f"{r['disorder_penalty']:.2f}", f"{r['ss_penalty']:.2f}",
                f"{r['sasa_penalty']:.2f}", f"{r['entropy']:.2f}",
                f"{r['extension']:.2f}", f"{r['disorder']:.2f}",
                r.get("secondary_structure", "NA"), r.get("sasa", "NA"),
            ])

    # 7. smoothed inverse-penalty plot ---------------------------
    unique = {}
    for r in ranked:
        unique.setdefault(r["position"], 1.0 / float(r["total_penalty"]))
    pos = np.array(sorted(unique), float)
    inv = np.array([unique[p] for p in pos], float)

    pos_s = np.linspace(pos.min(), pos.max(), 300)
    inv_s = make_interp_spline(pos, inv, k=3)(pos_s)

    plt.figure(figsize=(8, 6))
    plt.plot(pos, inv, "o", alpha=.35, label="raw")
    plt.plot(pos_s, inv_s, "-", label="spline")
    plt.xlabel("Position"); plt.ylabel("1 / TotalPenalty")
    plt.title(f"{uid}: inverse penalty profile")
    plt.grid(True); plt.legend()
    plt.savefig(f"img/{prefix}_inv_penalty.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"→ {csv_out}, img/{prefix}_inv_penalty.png")

# ------------ batch driver ------------
def main():
    if not pathlib.Path(CSV_MAP).exists():
        sys.exit(f"{CSV_MAP} not found")

    rows = [
        r for r in csv.DictReader(open(CSV_MAP))
        if r["uniprot_id"] and r["uniprot_id"] != "NA"
    ]
    if not rows:
        sys.exit("no UniProt IDs to process")

    for row in tqdm(rows, desc="batch"):
        uid      = row["uniprot_id"]
        pdb_path = PDB_DIR / f"{uid}.pdb"
        if not pdb_path.exists():
            tqdm.write(f"⚠ {uid}: PDB missing → skip")
            continue
        try:
            analyse(uid, pdb_path)
        except Exception as e:
            tqdm.write(f"💥 {uid}: {type(e).__name__}: {e}")
        time.sleep(0.1)   # gentle throttle

if __name__ == "__main__":
    main()

