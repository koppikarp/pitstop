#!/usr/bin/env python3
"""
download_alphafold_pdbs.py
- expects gene2uniprot.csv (gene_symbol,uniprot_id) in the cwd
- downloads AlphaFold PDBs into ./alphafold_pdbs/<UniProt>.pdb
"""
import csv, pathlib, requests, time
from tqdm import tqdm          # pip install tqdm

CSV_IN   = "gene2uniprot.csv"
OUT_DIR  = pathlib.Path("alphafold_pdbs")
BASE_URL = "https://alphafold.ebi.ac.uk/files"

# try v4, then v3, then v2 just in case
VERSIONS = ("v4", "v3", "v2")

def url_for(uid, ver):
    return f"{BASE_URL}/AF-{uid}-F1-model_{ver}.pdb"

def fetch(uid):
    for v in VERSIONS:
        url = url_for(uid, v)
        r   = requests.get(url, stream=True, timeout=30)
        if r.status_code == 200:
            return url, r
    return None, None  # nothing found

def main():
    OUT_DIR.mkdir(exist_ok=True)
    with open(CSV_IN) as fh:
        reader = csv.DictReader(fh)
        uids   = sorted({row["uniprot_id"] for row in reader if row["uniprot_id"] != "NA"})

    for uid in tqdm(uids, desc="downloading PDBs"):
        dest = OUT_DIR / f"{uid}.pdb"
        if dest.exists():
            continue  # already done

        url, resp = fetch(uid)
        if resp:
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            tqdm.write(f"✓ {uid} from {url.rsplit('/',1)[-1]}")
        else:
            tqdm.write(f"⚠ {uid}: no AlphaFold model")

        time.sleep(0.1)  # be civil

    print("all done.")

if __name__ == "__main__":
    main()
