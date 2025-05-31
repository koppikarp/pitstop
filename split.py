#!/usr/bin/env python3
"""
split_gene2uniprot.py
splits gene2uniprot.csv into 10 header-retaining parts: gene2up_part_00.csv ... gene2up_part_09.csv
"""

import csv, os

input_file = "gene2uniprot.csv"
prefix = "gene2uniprot_part_"

with open(input_file) as f:
    rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]

chunk_size = (len(data) + 9) // 10  # ceil divide
os.makedirs("splits", exist_ok=True)

for i in range(10):
    chunk = data[i * chunk_size:(i + 1) * chunk_size]
    out_path = f"splits/{prefix}{i:02d}.csv"
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(header)
        writer.writerows(chunk)

print("done splitting into splits/gene2uniprot_part_00.csv through _09.csv")

