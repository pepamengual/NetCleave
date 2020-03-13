from collections import Counter

def random_model_uniprot_collections(uniprot_data):
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
