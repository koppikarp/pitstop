#!/usr/bin/env python3
"""
annotations.py

Queries InterProScan (EBI) to retrieve domain, TM, signal peptide,
and PTM-like regions from a protein sequence.

Requires 'requests' to be installed: pip install requests

Example usage in your main pipeline:
  domain_ranges, tm_ranges, signal_pep_ranges, ptm_positions = get_domain_tm_signal_ptm(sequence)
"""

import subprocess
import json
import os
import requests

def get_tm_sp(sequence):
    # Write the sequence to a temporary FASTA file
    temp_filename = "temp_seq.fasta"
    with open(temp_filename, "w") as f:
        f.write(">query\n" + sequence + "\n")
    
    # Call the Perl script with the --json flag
    result = subprocess.run(
        ["perl", "phobius/phoutput.pl", "--json", temp_filename],
        stdout=subprocess.PIPE,
        check=True
    )
    # Remove the temporary file
    os.remove(temp_filename)
    
    # Decode and parse the JSON output
    annotations = json.loads(result.stdout.decode("utf-8"))
    
    # Assuming your Perl script returns a list with one dictionary:
    entry = annotations[0]
    tm_ranges = entry.get("tm_ranges", [])
    signal_pep_ranges = entry.get("signal_peptide_ranges", [])
    
    return tm_ranges, signal_pep_ranges

def get_ptm_positions(sequence):
    modeloptions = ["Phosphoserine_Phosphothreonine",
                "Phosphotyrosine",
                "N-linked_glycosylation",
                "O-linked_glycosylation",
                "Ubiquitination",
                "SUMOylation",
                "N6-acetyllysine",
                "Methylarginine",
                "Methyllysine",
                "Pyrrolidone_carboxylic_acid",
                "S-palmitoyl_cysteine",
                "Hydroxyproline",
                "Hydroxylysine"]

    model=modeloptions[0]+";"+modeloptions[4] #for multiple models
    url = "http://api.musite.net/musitedeep/"+model+"/"+sequence
    blasturl = "http://api.musite.net/blast/"+model+"/"+sequence
    myResponse = requests.get(url)
    if(myResponse.ok):
        # In this Example, jData are prediction results from MusiteDeep predictor
        data = json.loads(myResponse.content.decode('utf-8'))
        if "Error" in data.keys(): 
            print(data["Error in retrieving PTM data"])
    else:
        myResponse.raise_for_status()
    """
    Parses the JSON-like dictionary and returns a list of residue positions
    where at least one modification score exceeds the specified cutoff.
    """
    positions = []
    cutoff = 0.5  # Define your cutoff here
    for result in data.get("Results", []):
        # Retrieve the modification scores string from the PTMscores key.
        mod_scores = result.get("PTMscores", "")
        # Split the string into individual modifications in case there are multiple,
        # using semicolon as a separator.
        for mod in mod_scores.split(';'):
            if not mod.strip():
                continue  # Skip empty parts if any.
            try:
                # Each modification should be in the format "ModificationName:score".
                mod_name, score_str = mod.split(':')
                score = float(score_str)
                # If the score exceeds the cutoff, record the position.
                if score > cutoff:
                    positions.append(int(result.get("Position")))
                    # Stop checking further modifications for this result.
                    break
            except ValueError:
                print(f"Error processing modification: {mod}")


    ptm_positions = sorted(set(positions))

    return ptm_positions