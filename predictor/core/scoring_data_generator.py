

def generating_scoring_data(proteasome_dictionary, export_path):
    """ proteasome_dictionary contains as keys (residue in one code) and values (list of large peptides)
        large peptides contain in N-terminous the peptide, and in C-terminous the 4 following residues in sequence of that protein
        Hence, prediction must be done using 7 residues:
            3 before last residue of peptide
            1 last residue of peptide
            3 after last residue of peptide
    """
    for residue, proteasome_peptides in proteasome_dictionary.items():
        for proteasome_peptide in proteasome_peptides:
            peptide = proteasome_peptide[:-3]
            c_term = proteasome_peptide[-3:-1]
            prediction_peptide = proteasome_peptide[-8:-1]
            print(proteasome_peptide, peptide, c_term)






