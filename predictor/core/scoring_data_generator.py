def generating_scoring_data(proteasome_dictionary, export_path):
    """ proteasome_dictionary contains as keys (residue in one code) and values (list of large peptides)
        large peptides contain in N-terminous the peptide, and in C-terminous the 4 following residues in sequence of that protein
        Hence, prediction must be done using 7 residues:
            3 before last residue of peptide
            1 last residue of peptide
            3 after last residue of peptide
    """
    export_data = {}
    for residue, proteasome_peptides in proteasome_dictionary.items():
        for proteasome_peptide in proteasome_peptides:
            peptide = proteasome_peptide[:-4]
            prediction_region = proteasome_peptide[-8:-1]
            export_data.setdefault(residue, {}).setdefault(peptide, prediction_region)
    export_file(export_data, export_path)

def export_file(export_data, export_path):
    with open(export_path, "w") as f:
        f.write("residue\tpeptide\tprediction_region\n")
        for residue, peptide_dict in export_data.items():
            for peptide, prediction_region in peptide_dict.items():
                f.write("{}\t{}\t{}\n".format(residue, peptide, prediction_region))
