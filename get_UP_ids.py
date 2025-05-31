#!/usr/bin/env python3
"""
get_UP_ids_resume.py
resumable UniProt ID mapper with retries
reads geneNames.txt
writes/appends to gene2uniprot.csv
skips already-completed entries
"""
import csv, time, pathlib, requests
from tqdm import tqdm

FILE_IN  = "geneNames.txt"
FILE_OUT = "gene2uniprot.csv"
BASE     = "https://rest.uniprot.org/uniprotkb/search"
MAX_RETRIES = 3

def lookup(symbol, reviewed=True):
    q = f'gene_exact:{symbol} AND organism_id:9606'
    if reviewed:
        q += ' AND reviewed:true'
    r = requests.get(BASE, params={
        "query": q,
        "fields": "accession",
        "format": "list",
        "size": "1"
    }, timeout=30)
    r.raise_for_status()
    return r.text.strip()

def get_done_genes():
    done = set()
    if pathlib.Path(FILE_OUT).exists():
        with open(FILE_OUT) as fh:
            reader = csv.DictReader(fh)
            done.update(row["gene_symbol"] for row in reader if row["uniprot_id"] != "NA")
    return done

def main():
    genes = [g.strip() for g in open(FILE_IN) if g.strip()]
    done  = get_done_genes()
    pending = [g for g in genes if g not in done]

    if not pathlib.Path(FILE_OUT).exists():
        with open(FILE_OUT, "w", newline="") as out:
            csv.writer(out).writerow(["gene_symbol", "uniprot_id"])

    failures = []

    with open(FILE_OUT, "a", newline="") as out:
        writer = csv.writer(out)
        for g in tqdm(pending, desc="mapping genes"):
            acc = None
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    acc = lookup(g) or lookup(g, reviewed=False) or "NA"
                    break
                except Exception as e:
                    tqdm.write(f"retry {attempt} for {g} ({type(e).__name__})")
                    time.sleep(1.5 * attempt)  # backoff

            if not acc:
                acc = "NA"
                failures.append(g)

            writer.writerow([g, acc])
            out.flush()
            time.sleep(0.2)  # throttle

    if failures:
        print(f"\n⚠ {len(failures)} genes failed to resolve after retries:")
        for g in failures:
            print("  ", g)
    else:
        print("\n✓ all done with no failures.")

if __name__ == "__main__":
    main()
