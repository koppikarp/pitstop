#!/usr/bin/env python3
"""
features.py

Compute Shannon entropy, predict disorder (aiupred), and now parse
secondary structure (STRIDE) + SASA (freesasa) for the user-provided PDB.
"""

import math
from collections import Counter
import subprocess
import freesasa
import numpy as np

def calculate_shannon_entropy(msa_sequences, ignore_gaps=True):
    if not msa_sequences:
        return []
    alignment_length = len(msa_sequences[0]) # all sequences should have the same length, first item is the query
    print(alignment_length)
    entropies = []
    for pos in range(alignment_length):
        column = [seq[pos] for seq in msa_sequences]
        if ignore_gaps:
            column = [res for res in column if res[0] != '-']
        if not column:
            entropies.append(0.0)
            continue
        counts = Counter(column)
        entropy = 0.0
        for aa, count in counts.items():
            p = count / len(column)
            entropy -= p * math.log2(p)
        entropies.append(entropy)
    return entropies
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

    # # test score change with artificial ext block
    # extendable_blocks.append({
    #                         'start': 110,
    #                         'end': 110,
    #                         'length': 10,
    #                         'seq_index': 6,
    #                     })

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

def predict_disordered_binding_regions(sequence):
    """
    Predict per-residue disorder using the aiupred library.
    Returns {pos: (aa, disorder_score, None)} with anchor_score=None.
    """
    from aiupred import aiupred_lib

    embedding_model, regression_model, device = aiupred_lib.init_models()
    disorder_scores = aiupred_lib.predict_disorder(sequence, embedding_model, regression_model, device)

    predictions = {}
    for i, score in enumerate(disorder_scores, start=1):
        aa = sequence[i-1]
        predictions[i] = (aa, float(score), None)
    return predictions

def get_secondary_structure_and_sasa(pdb_file, stride_executable="stride", chain_id="A"):
    """
    Uses STRIDE for secondary structure, freesasa for SASA.

    Returns a dict:  { (chain, residue_number) : (ss_code, sasa_value) }

    Example SS codes from STRIDE:
      H = alpha-helix, E = beta-strand, C = coil/other, ...
    """
    # --- 1. Parse secondary structure with STRIDE ---
    # Run stride on the PDB
    cmd = [stride_executable, pdb_file]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print("Error running STRIDE:", e)
        return {}

    ss_sasa = {}

    # STRIDE lines for residue annotations typically start with "ASG"
    # Format example:
    # ASG  MET A   1    H ...
    # ^^^  ^^^ ^   ^   ^ 
    #  1   2   3   4   5
    # We'll parse the residue number, chain, and SS code.
    # Example fix inside get_secondary_structure_and_sasa():
    for line in result.stdout.splitlines():
        if not line.startswith("ASG "):
            continue
        print(line)
        tokens = line.split()

        # tokens[3] = '1' (res num)
        # tokens[4] = '1' (some extra field, e.g. sequence index)
        # tokens[5] = 'C' (the single-letter SS code)
        # tokens[6] = 'Coil' (the descriptive name)

        if len(tokens) < 6:
            continue  # skip malformed lines
        
        chain = tokens[2]               # e.g. 'A'
        res_num = int(tokens[3])        # e.g. 1
        ss_code = tokens[5]            # e.g. 'C' or 'E' or 'H'
        
        # Store in your dict:
        ss_sasa[(chain, res_num)] = (ss_code, None)
    print('ss_sasa: ', ss_sasa)

    # --- 2. Parse SASA with freesasa ---
    structure = freesasa.Structure(pdb_file)
    result_freesasa = freesasa.calc(structure)
    residue_areas = result_freesasa.residueAreas()
    print('residue areas: ', residue_areas)

    # residue_areas is a dict: chain -> {resnum: ResidueArea}, etc.
    for chain, residues_dict in residue_areas.items():
        for res_num, res_area_obj in residues_dict.items():
            sasa_val = res_area_obj.total

            res_num_int = int(res_num)
            # If (chain, res_num) already in ss_sasa => update. 
            # If not found, add it anyway.
            if (chain, res_num) in ss_sasa:
                old_ss, _ = ss_sasa[(chain, res_num)]
                ss_sasa[(chain, res_num)] = (old_ss, sasa_val)
            else:
                ss_sasa[(chain, res_num)] = (None, sasa_val)

    return ss_sasa

def parse_ss_sasa_for_chain(ss_sasa_dict, chain_id='A'):
    # First gather all entries (rnum, ss, sasa)
    entries = []
    for (c, rnum), (ss, sasa) in ss_sasa_dict.items():
        if c == chain_id:
            # Convert rnum to int if it might be string
            rnum_int = int(rnum)
            entries.append((rnum_int, ss, sasa))

    # Now merge any duplicates on rnum
    merged_map = {}  # rnum -> (ss, sasa)
    for (rnum_int, ss, sasa) in entries:
        if rnum_int not in merged_map:
            merged_map[rnum_int] = [None, None]  # [ss, sasa]
        # If we see a non-None ss, store it
        if ss is not None:
            merged_map[rnum_int][0] = ss
        # If we see a non-None sasa, store it
        if sasa is not None:
            merged_map[rnum_int][1] = sasa

    # Build final sorted list
    final_list = []
    for rnum_int in sorted(merged_map.keys()):
        ss_val, sasa_val = merged_map[rnum_int]
        final_list.append((ss_val, sasa_val))  # e.g. ('C', 250.45)
    return final_list

def combine_features(query_alignment, cleavage_pos, entropy_list, extension_list, disorder_dict, ss_sasa_list=None):
    """
    Build a list of feature dicts for each residue in the aligned query sequence.
    We'll skip positions that are '-' in the alignment.
    """
    results = []
    query_pos = 0
    if cleavage_pos[0] != -1:
        signalp_loc = cleavage_pos[0]
    else:
        signalp_loc = 0
    for i, char in enumerate(query_alignment):
        if char == '-':
            continue
        query_pos += 1

        # Basic features
        feat = {
            "position": query_pos,
            "residue": char,
            "entropy": entropy_list[i] if i < len(entropy_list) else 0.0,
            "extension": extension_list[i] if i < len(extension_list) else 0.0
        }

        # Disorder from aiupred
        if query_pos in disorder_dict:
            _, dscore, ascore = disorder_dict[query_pos]
            feat["disorder"] = dscore
            feat["anchor"] = ascore
        else:
            feat["disorder"] = 0.0
            feat["anchor"] = None
        print(ss_sasa_list)
        # Secondary structure + SASA if provided
        if ss_sasa_list and (query_pos - signalp_loc) <= len(ss_sasa_list) and query_pos > signalp_loc: # need to add the buffer in case of signal peptide cleavage
            print('query_pos ', query_pos)
            print('sasa list position ', query_pos - signalp_loc)
            print('\n')
            ss, sasa = ss_sasa_list[query_pos - signalp_loc - 1] # get zero positioning by python list for first position after sp
            feat["secondary_structure"] = ss  # e.g. 'H', 'E', 'C', None
            feat["sasa"] = sasa              # numeric
        else:
            feat["secondary_structure"] = None
            feat["sasa"] = None

        results.append(feat)
    print(results)
    return results
