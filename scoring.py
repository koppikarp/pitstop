#!/usr/bin/env python3
"""
scoring.py

Now also considers secondary_structure & sasa in the final penalty.
"""
import numpy as np

def mark_excluded_positions(length, ranges):
    excluded = set()
    for (s, e) in ranges:
        for i in range(s, e+1):
            if 1 <= i <= length:
                excluded.add(i)
    return excluded

def assign_domain_penalties(length, domain_ranges, penalty=1.0):
    d = {i: 0.0 for i in range(1, length+1)}
    for (s, e) in domain_ranges:
        for i in range(s, e+1):
            if i in d:
                d[i] = penalty
    return d

def assign_ptm_penalties(length, ptm_positions, base_penalty=1.0, distance_decay=True):
    d = {i: 0.0 for i in range(1, length+1)}
    if not ptm_positions:
        return d

    if not distance_decay:
        for p in ptm_positions:
            if 1 <= p <= length:
                d[p] = base_penalty
    else:
        cutoff = 3
        for i in range(1, length+1):
            dist = min(abs(i - p) for p in ptm_positions)
            if dist <= cutoff:
                d[i] = base_penalty / (1 + dist)
    return d

def combine_all_scores(query_seq, combined_features, ptm_penalty_dict, exclude_positions, excluded_signal, cleavage_pos):
    # First, compute the maximum SASA from the features that are not excluded.
    valid_sasa = [
        feat["sasa"] 
        for feat in combined_features 
        if feat.get("sasa") is not None and feat["position"] not in excluded_signal
    ]
    # If no valid SASA values are found, fall back to a default value (e.g., 250)
    max_possible_sasa = max(valid_sasa) if valid_sasa else 250.0

    valid_entropy = [
        feat["entropy"] 
        for feat in combined_features 
        if feat.get("entropy") is not None and feat["position"] not in excluded_signal
    ]

    # If no valid SASA values are found, fall back to a default value (e.g., 250)
    max_possible_entropy = max(valid_entropy) if valid_entropy else 4.32

    results = []
    for feat in combined_features:
        i = feat["position"]
        if i in excluded_signal:
            continue

        # Existing penalty contributions
        ptm_pen = ptm_penalty_dict.get(i, 0.0)
        entropy = feat.get("entropy", 0.0) or 0.0
        extension = feat.get("extension", 0.0) or 0.0
        disorder = feat.get("disorder", 0.0) or 0.0

        # Scoring:
        conservation_pen = max(0.0, 1.0 - (entropy/max_possible_entropy))  # penalize strong conservation
        ext_pen = max(0.0, 1.0 - extension)  # penalize weaj extension
        disorder_pen = 1.0 - disorder # prefer disordered => lower penalty if disorder is high

        # SS & SASA values
        sasa_pen = 0.0
        ss_pen = 0.0
        if i >= cleavage_pos[0]:
            ss = feat.get("secondary_structure", None)
            sasa = feat.get("sasa", None)

            # Normalize SASA using the maximum SASA from all valid features
            if sasa is not None:
                normalized_sasa = min(sasa / max_possible_sasa, 1.0)
                # Higher SASA means more exposure; here we penalize buried (lower SASA) residues.
                sasa_pen = 1.0 - normalized_sasa

            # Penalize secondary structure based on the structure type.
            if ss == "H":
                ss_pen = 1
            elif ss == "E":
                ss_pen = 0.9
            elif ss == "G":
                ss_pen = 0.7
            elif ss == "I":
                ss_pen = 0.6
            elif ss == "T":
                ss_pen = 0.5
            elif ss == "S":
                ss_pen = 0.3
            elif ss == "C":
                ss_pen = 0.0

        # weights
        w1 = 0.4 # extension is not often a deciding factor, not sure how much we can take from evolution. kinda a proxy for disorder in a way (2/10)
        w2 = 1.4 # don’t wanna ruin key regulatory mods, but distance decay already integrates this (7/10)
        w3 = 1.0 # useful for flagging conservation, but less directly tied to insertion impact (5/10)
        w4 = 1.2 # flexible, forgiving spots for a tag (6/10)
        w5 = 1.6 # inserting in a helix or sheet can scramble local structure (8/10)
        w6 = 1.8 # if the tag’s buried, it’s basically invisible and disruptive (9/10)

        total_pen = (w1 * ext_pen) + (w2 * ptm_pen) + (w3 * conservation_pen) + (w4 * disorder_pen) + (w5 * ss_pen) + (w6 * sasa_pen)
        
        if i in exclude_positions:
            continue

        results.append({
            "position": i,
            "residue": feat["residue"],
            "total_penalty": total_pen,
            "ptm_penalty": ptm_pen,
            "conservation_penalty": conservation_pen,
            "disorder_penalty": disorder_pen,
            "ss_penalty": ss_pen,
            "sasa_penalty": sasa_pen,
            "entropy": entropy,
            "extension": extension,
            "disorder": disorder,
            "secondary_structure": ss,
            "sasa": sasa
        })

    # Sort ascending => "least-worst"
    results.sort(key=lambda x: x["total_penalty"])
    return results
