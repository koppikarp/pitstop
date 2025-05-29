from Bio.PDB import MMCIFParser, PDBIO

# Define your input and output file names
cif_file = "q9num4.cif"   # Replace with your mmCIF file path
pdb_file = "q9num4.pdb"  # Replace with your desired PDB file path

# Create an MMCIF parser object; QUIET=True suppresses warnings
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("structure", cif_file)

# Create a PDBIO object and set the structure to be saved
io = PDBIO()
io.set_structure(structure)
io.save(pdb_file)
