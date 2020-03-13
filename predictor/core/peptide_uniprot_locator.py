def locate_peptides(ms_data, uniprot_data):
    """
    Looks for adjacent C-terminal sequence for each peptide
    Remove peptides where some special character appears, that is not an standard amino acid
    Returns a list of large peptides, and a dictionary where:
        key = C-terminal residue of the peptide (IEDB)
        values = a list of peptides
    """
    print("Locating IEDB peptides into Uniprot sequences...")
    amino_acids_possible_set = set_creator_of_amino_acids("ACDEFGHIKLMNPQRSTVWY")
    adjacent_lenght = 4
    data_dict = {}
    for uniprot_id, peptide_set in ms_data.items():
        if uniprot_id in uniprot_data:
            for peptide in peptide_set:
                if peptide in uniprot_data[uniprot_id]:
                    adjacent_sequences = uniprot_data[uniprot_id].split(peptide)
                    post_sequence = adjacent_sequences[1][:adjacent_lenght]
                    if len(post_sequence) == adjacent_lenght:
                        large_peptide = "".join(peptide + post_sequence)
                        large_peptide_set = set(large_peptide)
                        if len(large_peptide_set.difference(amino_acids_possible_set)) == 0:
                            data_dict.setdefault(peptide[-1], []).append(large_peptide)
    return data_dict

def set_creator_of_amino_acids(amino_acids_possible):
    amino_acids_possible_set = set()
    for amino_acid in amino_acids_possible:
        amino_acids_possible_set.add(amino_acid)
    return amino_acids_possible_set
