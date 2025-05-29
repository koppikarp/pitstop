#!/usr/bin/env python3
"""
msa.py

Fetch a sequence from UniProt or user input, run BLAST, and build a trivial MSA.
"""

import sys
import io
import requests
from Bio import SeqIO, Entrez
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIWWW, NCBIXML
from skbio import Protein, TabularMSA
from skbio.io import read
import subprocess

Entrez.email = "pratik.koppikar@nih.gov"  # Replace with your actual email

def fetch_sequence_from_uniprot(uniprot_id):
    """Fetch the FASTA sequence from UniProt for a given UniProt ID."""
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url, verify=False) # temporary solution, disables verification
    if response.status_code == 200:
        fasta_data = response.text
        record = SeqIO.read(io.StringIO(fasta_data), "fasta")
        return record
    else:
        print(f"Error fetching sequence for UniProt ID {uniprot_id} (status code {response.status_code}).")
        sys.exit(1)

def blast_search(query_sequence):
    """Run a BLASTP search on NCBI’s nr database using the given query sequence."""
    print("Running BLAST search (this may take several minutes)...")
    result_handle = NCBIWWW.qblast("blastp", "nr", query_sequence)
    blast_record = NCBIXML.read(result_handle)
    return blast_record

def fetch_full_sequence(accession):
    """Retrieve the full FASTA sequence from NCBI Protein using an accession number."""
    try:
        handle = Entrez.efetch(db="protein", id=accession, rettype="fasta", retmode="text")
        record = SeqIO.read(handle, "fasta")
        handle.close()
        return record
    except Exception as e:
        print(f"Error fetching sequence for accession {accession}: {e}")
        return None

def trivial_msa(sequences):
    """
    Construct a trivial multiple sequence alignment by right-padding sequences
    with '-' characters so that all sequences have the length of the longest one.
    """
    max_len = max(len(seq) for seq in sequences)
    aligned_sequences = []
    for seq in sequences:
        padded_seq = seq.ljust(max_len, '-')
        aligned_sequences.append(Protein(padded_seq))
    msa = TabularMSA(aligned_sequences)
    return msa

def mafft_msa(sequences):
    """
    Construct a multiple sequence alignment by MAFFT.
    """
    # create SeqRecord objects with generic IDs
    seq_records = [SeqRecord(Seq(seq), id=f"seq{i+1}", description="") for i, seq in enumerate(sequences)]

    # write to fasta file
    input_fasta = "blast_records.fasta"
    with open(input_fasta, "w") as f:
        SeqIO.write(seq_records, f, "fasta")

    output_fasta = "mafft_msa.fasta"

    subprocess.run(f"mafft {input_fasta} > {output_fasta}", shell=True, check=True)

    # read aligned protein sequences from fasta file
    aligned_proteins = list(read(output_fasta, format="fasta", constructor=Protein))
    msa = TabularMSA(aligned_proteins)
    return msa

def perform_msa(query_record):
    """
    Given a query record, performs a BLAST search and constructs a trivial MSA.

    Returns:
      msa_result: a TabularMSA object
      hit_records: list of SeqRecord objects (including the query).
    """
    blast_record = blast_search(str(query_record.seq))
    hit_records = [query_record]
    for alignment in blast_record.alignments:
        accession = alignment.accession
        record = fetch_full_sequence(accession)
        if record:
            hit_records.append(record)
            print(f"Retrieved sequence for accession {accession}")
    print(f"Total of {len(hit_records)} sequences for MSA.")
    sequences = [str(r.seq) for r in hit_records]
    msa_result = mafft_msa(sequences)
    return msa_result, hit_records
