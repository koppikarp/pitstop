import math
from skbio import TabularMSA

import math
from collections import Counter
import subprocess
import freesasa
import numpy as np

import sys
import csv
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

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

def calculate_extendable_scores(msa_sequences):
    """
    Calculate per-residue extendability scores for the query sequence (first sequence in msa_sequences).
    
    The score is computed for each residue (non-gap) position in the query sequence using the following idea:
      - For each residue, find the nearest extendable region (a contiguous block where the query has a gap,
        block length >= 2) on the left and right.
      - For each side, compute a score:
          s = exp(-alpha * distance) * (1 - exp(-beta * block_length)) * conservation
        where:
          distance (d) is the number of columns between the residue and the extendable block,
          block_length (L) is the length of the extendable region,
          conservation (C) is the fraction of other sequences that have a residue (non-gap) in all columns of that block.
      - Combine the left (s_L) and right (s_R) scores using:
          S = min((s_L + s_R)/2 + lambda * s_L * s_R, 1)
        so that having good extendability on both sides yields a bonus.
    
    Parameters:
      msa_sequences (TabularMSA): MSA with the query as the first sequence.
      
    Returns:
      A list of extendability scores (floats between 0 and 1) for each column of the query sequence.
      For positions where the query has a gap, the score is set to 0.
    """
    
    import math

    # Parameter definitions (adjustable)
    alpha = 1.05   # decay rate for distance: higher alpha -> faster decay with distance
    beta = 0.5    # saturation rate for region length: higher beta -> faster saturation
    synergy_lambda = 0.25  # synergy bonus factor for having extendability on both sides
    
    # Get the query sequence (first sequence) as a string.
    query_seq = str(msa_sequences[0])
    aln_length = len(query_seq)
    
    # Preprocess: Identify all extendable regions (contiguous gap blocks in query of length >= 2)
    extendable_blocks = []
    aln_length = len(msa_sequences[0])

    i = 0
    while i < aln_length:
        if query_seq[i] == '-':
            start = i
            while i < aln_length and query_seq[i] == '-':
                i += 1
            end = i - 1
            block_length = end - start + 1
            if block_length >= 2:
                # Check if at least one sequence has a continuous aa segment of >=2 residues within this block
                valid_seqs = []
                for seq_idx, seq in enumerate(msa_sequences[1:], start=1):
                    segment = str(seq[start:end + 1])
                    segments = segment.split('-')
                    has_valid_segment = any(len(seg) >= 2 for seg in segments)
                    if has_valid_segment:
                        valid_segment_length = max(len(seg) for seg in segments)
                        valid_segment_start = segment.find(max(segments, key=len))
                        extendable_blocks.append({
                            'start': start + valid_segment_start,
                            'end': start + valid_segment_start + valid_segment_length - 1,
                            'length': valid_segment_length,
                            'seq_index': seq_idx,
                        })
        else:
            i += 1

    # test score change with artificial ext block
    extendable_blocks.append({
                            'start': 110,
                            'end': 110,
                            'length': 10,
                            'seq_index': 6,
                        })
    # Recalculate conservation for each block separately
    for block in extendable_blocks:
        block_start = block['start']
        block_end = block['end']
        block_len = block['length']
        count = 0
        total = len(msa_sequences) - 1

        for seq in msa_sequences[1:]:
            segment = seq[block_start:block_end + 1]
            if '-' not in segment:
                count += 1  # fully conserved in this block

        block['conservation'] = count / total if total > 0 else 0

    # (Optional) print to verify correctness
    print(f"Validated extendable blocks: {extendable_blocks}")

    # For each residue (non-gap) in the query, calculate extendability score.
    scores = [0.0] * aln_length  # default score for each column; will remain 0 if query has gap.
    
    # Helper function: given a list of blocks and a position, find the nearest block in a given direction.
    # direction = 'left' or 'right'
    def find_nearest_block(pos, direction):
        nearest = None
        min_distance = None
        if direction == 'left':
            for block in extendable_blocks:
                # Only consider blocks that are entirely to the left of the residue.
                if block['end'] < pos:
                    # Distance: number of columns between residue and block's right end.
                    d = pos - block['end'] - 1
                    if min_distance is None or d < min_distance:
                        min_distance = d
                        nearest = block
        elif direction == 'right':
            for block in extendable_blocks:
                # Only consider blocks that are entirely to the right of the residue.
                if block['start'] > pos:
                    # Distance: number of columns between block's left start and the residue.
                    d = block['start'] - pos - 1
                    if min_distance is None or d < min_distance:
                        min_distance = d
                        nearest = block
        return nearest, min_distance if min_distance is not None else None
    
    # Calculate score for each column in the query that is a residue.
    for pos in range(aln_length):
        if query_seq[pos] == '-':
            # If the query has a gap at this position, score remains 0.
            continue
        
        # For each side, look for the nearest extendable block.
        left_block, left_distance = find_nearest_block(pos, 'left')
        right_block, right_distance = find_nearest_block(pos, 'right')
        
        # Compute left score
        if left_block is not None and left_distance is not None:
            s_left = max(0, 1 - (left_distance/len(query_seq))) * (1 - math.exp(-beta * left_block['length'])) * left_block['conservation']
        else:
            s_left = 0.0
        
        # Compute right score
        if right_block is not None and right_distance is not None:
            # divide distance by length of query sequence to normalize decay based on length
            s_right = max(0, 1 - (right_distance/len(query_seq))) * (1 - math.exp(-beta * right_block['length'])) * right_block['conservation']
        else:
            s_right = 0.0
        
        # Combine the two side scores using a synergy-boosted approach.
        combined = (s_left + s_right) / 2 + (synergy_lambda * s_left * s_right)
        # Ensure the combined score does not exceed 1.
        scores[pos] = min(combined, 1.0)
    print(scores)
    extension = np.array(scores)
    min_score = np.min(extension)
    max_score = np.max(extension)
    if np.isclose(max(extension), 0):
        # Avoid division by zero; return zeros.
        return [0.0 for _ in extension]

    # Min-max normalization
    normalized_scores = ((extension - np.min(extension)) / (max(extension) - np.min(extension))).tolist()
    print(normalized_scores)

    return normalized_scores

# Example usage (assuming msa_sequences is a TabularMSA instance):
fasta = "mafft_msa.fasta"
# read aligned protein sequences from fasta file
aligned_proteins = list(read(fasta, format="fasta", constructor=Protein))
msa = TabularMSA(aligned_proteins)
scores = calculate_extendable_scores(msa)
#print(scores)
