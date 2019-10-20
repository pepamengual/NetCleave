def random_model_all_peptides(data):
    amino_acid_count = 0
    data_dictionary = {}
    for peptide in data:
        for amino_acid in peptide:
            amino_acid_count += 1
            data_dictionary.setdefault(amino_acid, 0)
            data_dictionary[amino_acid] += 1
    data_frequency_dictionary = {}
    
    f = 0
    for amino_acid, counts in data_dictionary.items():
        frequency = counts / amino_acid_count
        f += frequency
        data_frequency_dictionary.setdefault(amino_acid, frequency)
    print(f)
    return data_frequency_dictionary


def random_model_uniprot(uniprot_data):
    data_dictionary = {}
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    amino_acid_count = 0
    for uniprot_id, peptide_sequence in uniprot_data.items():
        for residue in peptide_sequence:
            if residue in amino_acids:
                data_dictionary.setdefault(residue, 0)
                data_dictionary[residue] += 1
                amino_acid_count += 1
    data_frequency_dictionary = {}
    for amino_acid, counts in data_dictionary.items():
        frequency = counts / amino_acid_count
        data_frequency_dictionary.setdefault(amino_acid, frequency)
    return data_frequency_dictionary
