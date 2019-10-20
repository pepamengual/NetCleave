def random_model_maker(data):
    data_dictionary = {}
    for peptide in data:
        for amino_acid in peptide:
            data_dictionary.setdefault(amino_acid, 0)
            data_dictionary[amino_acid] += 1
    data_frequency_dictionary = {}
    for amino_acid, counts in data_dictionary.items():
        frequency = counts / len(data)
        data_frequency_dictionary.setdefault(amino_acid, frequency)
    return data_frequency_dictionary
