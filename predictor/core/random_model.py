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

def random_model_uniprot_collections(uniprot_data):
    from collections import Counter
    single_string_proteome = "".join(uniprot_data.values())
    all_counts = Counter(single_string_proteome)
    data_dictionary = {}
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    for amino_acid in amino_acids:
        data_dictionary.setdefault(amino_acid, all_counts[amino_acid])
    amino_acid_count = sum(data_dictionary.values())
    data_frequency_dictionary = {}
    for amino_acid, counts in data_dictionary.items():
        frequency = counts / amino_acid_count
        data_frequency_dictionary.setdefault(amino_acid, frequency)
    return data_frequency_dictionary
