#!/usr/bin/env python3
"""
main.py
"""

import sys
import csv
import os
import ast
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import msa
import features
import annotations
import scoring
import subprocess

def get_query_sequence():
    print("You will need either your protein sequence or a UniProt ID, as well as the corresponding PDB file for that sequence. \nPlease use AlphaFold Server to generate the PDB if you do not already have one.\n")

    print("Choose input type:")
    print("1. Provide an amino acid sequence")
    print("2. Provide a UniProt ID")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        seq_input = input("Enter your amino acid sequence: ").strip()
        query_record = SeqRecord(Seq(seq_input), id="Query", description="User provided sequence")
    elif choice == "2":
        uniprot_id = input("Enter the UniProt ID: ").strip()
        query_record = msa.fetch_sequence_from_uniprot(uniprot_id)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    return query_record

def main():
    # 1. Get query sequence and pdb file
    query_record = get_query_sequence()
    full_seq = str(query_record.seq)
    SeqIO.write(query_record, "query.fasta", "fasta")
    pdb_file = input("Enter the path to your PDB file: ").strip()
    pdb_name = pdb_file.split("/")[-1].split(".")[0]
    print(f"Using PDB file: {pdb_name}")

    # 1.5. Check if the PDB file exists
    if not os.path.isfile(pdb_file):
        print(f"Error: The file {pdb_file} does not exist.")
        sys.exit(1)

    # 1.9. Modify structural file if signal peptide present
    command = [
        "python3",
        "signalp6_fast/signalp-6-package/signalp/predict.py",
        "--fastafile", "query.fasta",
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    output_str = result.stdout.strip()
    print("result ", result)
    print("Subprocess raw stdout:", repr(result.stdout))
    print("Subprocess output:", output_str, type(output_str))

    cleavage_pos = ast.literal_eval(result.stdout.strip())
    # cleavage_pos = [27]

    if cleavage_pos[0] != -1:
        cleavage_index = cleavage_pos[0]
        print(f"Signal peptide detected. Cleavage position: {cleavage_index}.")
        print("Modifying the PDB file to remove atoms/residues before the cleavage site...")

        new_lines = []
        with open(pdb_file, 'r') as infile:
            for line in infile:
                # Process ATOM/HETATM lines that contain coordinate/residue data.
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # The residue sequence number is in columns 23-26.
                    resnum_str = line[22:26].strip()
                    try:
                        resnum = int(resnum_str)
                    except ValueError:
                        # If we can't parse the residue number, skip this line.
                        continue

                    # Only include lines for residues after the cleavage position.
                    # (Adjust the comparison if you want to include the cleavage residue itself.)
                    if resnum > cleavage_index:
                        new_lines.append(line)
                else:
                    # Optionally keep non-ATOM lines (like headers) or skip them.
                    new_lines.append(line)

        mature_pdb_file = f"{pdb_name}_mature.pdb"
        with open(mature_pdb_file, 'w') as outfile:
            outfile.writelines(new_lines)

        sp_present = True
        print(f"Modified PDB file without SP to be used written to {mature_pdb_file}.")
    else:
        sp_present = False
        print("No signal peptide detected. PDB file remains unmodified.")

    # 2. Perform MSA
    msa_result, hits = msa.perform_msa(query_record)
    msa_seqs = [str(seq) for seq in msa_result]
    if not msa_seqs:
        print("No MSA sequences found, aborting.")
        sys.exit(1)
    query_alignment = msa_seqs[0]

    
    # 3. Compute features: Shannon entropy + AIUPred disorder
    entropy_list = features.calculate_shannon_entropy(msa_seqs)
    extension_list = features.calculate_extendable_scores(msa_seqs)
    disorder_dict = features.predict_disordered_binding_regions(full_seq)

    # 4. Ask about PDB for STRIDE + freesasa
    ss_sasa_list = None
    
    stride_path = "stride/stride"  # or the path to your stride executable
    chain_id = "A"          # assume we want chain A

    # ss_sasa_dict = features.get_secondary_structure_and_sasa(pdb_file, stride_executable=stride_path, chain_id=chain_id)
    if sp_present:
        ss_sasa_dict = features.get_secondary_structure_and_sasa(mature_pdb_file, stride_executable=stride_path, chain_id=chain_id)
    else:    
        ss_sasa_dict = features.get_secondary_structure_and_sasa(pdb_file, stride_executable=stride_path, chain_id=chain_id)
    ss_sasa_list = features.parse_ss_sasa_for_chain(ss_sasa_dict, chain_id=chain_id)
    print(f"len(query_alignment): {len(query_alignment)}", query_alignment)
    print(f"len(ss_sasa_list): {len(ss_sasa_list)}", ss_sasa_list)
    
    # 5. Combine all features into a single list
    combined_feats = features.combine_features(query_alignment, cleavage_pos, entropy_list, extension_list, disorder_dict, ss_sasa_list)

    # 6. Annotate with InterProScan
    print("\nSubmitting to Phobius for sp/tm. Please wait...")
    tm_ranges, signal_pep_ranges = annotations.get_tm_sp(full_seq)

    print("\nSubmitting to MusiteDeep for ptm. Please hold...")
    ptm_positions = annotations.get_ptm_positions(full_seq)
    print("Annotation complete.")
    print(f"Transmembrane: {tm_ranges}")
    print(f"Signal peptides: {signal_pep_ranges}")
    print(f"PTM positions: {ptm_positions}")

    # 7. Build domain + PTM penalties
    length = len(full_seq)
    ptm_penalties = scoring.assign_ptm_penalties(length, ptm_positions, base_penalty=1.0, distance_decay=True)

    # 8. Exclude signal + TM regions
    excluded_signal = scoring.mark_excluded_positions(length, signal_pep_ranges)
    excluded_tm = scoring.mark_excluded_positions(length, tm_ranges)
    exclude_positions = excluded_signal.union(excluded_tm)

    # 9. Combine everything => final ranking
    final_scores = scoring.combine_all_scores(
        full_seq,
        combined_feats,
        ptm_penalties,
        exclude_positions,
        excluded_signal,
        cleavage_pos
    )

    # 10. Print top hits
    print("\n=== Top 10 'least-worst' insertion sites ===")
    print("Pos\tAA\tTotPen\tPTMPen\tConPen\tDisPen\tSSPen\tSASAPen\tEntropy\tExtension\tDisorder\tSS\tSASA")
    for row in final_scores[:10]:
        print(f"{row['position']}\t{row['residue']}\t"
              f"{row['total_penalty']:.2f}\t"
              f"{row['ptm_penalty']:.2f}\t"
              f"{row['conservation_penalty']:.2f}\t{row['disorder_penalty']:.2f}\t"
              f"{row['ss_penalty']:.2f}\t{row['sasa_penalty']:.2f}\t"
              f"{row['entropy']:.2f}\t{row['extension']:.2f}\t{row['disorder']:.2f}\t"
              f"{row.get('secondary_structure','NA')}\t"
              f"{row.get('sasa','NA')}")

    #11. Plot a position-wise plot of tagging scores
    positions = [row["position"] for row in final_scores]
    inverse_penalties = [1.0 / float(row["total_penalty"]) for row in final_scores]

    # Sort by position
    sorted_data = sorted(zip(positions, inverse_penalties), key=lambda x: x[0])
    positions_sorted, inv_penalties_sorted = zip(*sorted_data)

    positions_sorted = np.array(positions_sorted, dtype=float)
    inv_penalties_sorted = np.array(inv_penalties_sorted, dtype=float)

    # Create a "denser" set of x-values for a smooth curve
    x_smooth = np.linspace(positions_sorted.min(), positions_sorted.max(), 300)

    # Use a cubic spline (k=3) to interpolate
    spline = make_interp_spline(positions_sorted, inv_penalties_sorted, k=3)
    inv_penalties_smooth = spline(x_smooth)

    plt.figure(figsize=(8, 6))

    # Optional: plot the original points (lightly) for reference
    plt.plot(positions_sorted, inv_penalties_sorted, 'o', alpha=0.4, label='Original data')

    # Plot the smoothed line
    plt.plot(x_smooth, inv_penalties_smooth, '-', label='Smoothed spline')

    plt.xlabel("Position")
    plt.ylabel("Inverse Total Penalty Score (1 / TotalPenalty)")
    plt.title("Positions vs. Inverse Total Penalty (Smoothed)")
    plt.grid(True)
    plt.legend()
    
    plt.savefig(f"{pdb_name}_ positions_vs_inverse_penalty_smoothed.png", dpi=300, bbox_inches="tight")

    # 12. Write all positions to CSV
    csv_filename = f"{pdb_name}_scores.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Position", "Residue", "TotalPenalty",
                         "PTMPenalty", "ConPen", "DisorderPen",
                         "SSPenalty", "SASAPenalty",
                         "Entropy", "Extension", "Disorder", "SecondaryStructure", "SASA"])
        for row in final_scores:
            writer.writerow([
                row["position"],
                row["residue"],
                f"{row['total_penalty']:.2f}",
                f"{row['ptm_penalty']:.2f}",
                f"{row['conservation_penalty']:.2f}",
                f"{row['disorder_penalty']:.2f}",
                f"{row['ss_penalty']:.2f}",
                f"{row['sasa_penalty']:.2f}",
                f"{row['entropy']:.2f}",
                f"{row['extension']:.2f}",
                f"{row['disorder']:.2f}",
                row.get("secondary_structure","NA"),
                row.get("sasa","NA")
            ])

    # Additional snippet to print N-terminal, C-terminal, and highest ranked non-terminal positions
    print("\n=== Terminal and Highest Ranked Non-terminal Positions ===")

    # Sort by position to identify terminals clearly
    final_scores_sorted_by_pos = sorted(final_scores, key=lambda x: x['position'])

    # N-terminus (first position)
    n_term = final_scores_sorted_by_pos[0]

    # C-terminus (last position)
    c_term = final_scores_sorted_by_pos[-1]

    # Highest ranked non-terminal (excluding first and last positions)
    non_terminal_scores = [row for row in final_scores if row['position'] not in [n_term['position'], c_term['position']]]
    highest_non_terminal = sorted(non_terminal_scores, key=lambda x: x['total_penalty'])[0]

    # Print formatted results
    print("Position Type\tPos\tAA\tTotal Penalty")
    print(f"N-Terminus\t{n_term['position']}\t{n_term['residue']}\t{n_term['total_penalty']:.2f}")
    print(f"C-Terminus\t{c_term['position']}\t{c_term['residue']}\t{c_term['total_penalty']:.2f}")
    print(f"Highest Non-terminal\t{highest_non_terminal['position']}\t{highest_non_terminal['residue']}\t{highest_non_terminal['total_penalty']:.2f}")

    print(f"\nAll scored positions written to {csv_filename}. \nPlease check for annotation accuracy!")

if __name__ == "__main__":
    main()
