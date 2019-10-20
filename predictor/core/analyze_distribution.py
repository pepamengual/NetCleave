
def distribution_analyzer(data, adjacent_lenght):
    data_dictionary = {}
    for peptide in data:
        for position, residue in enumerate(peptide):
            if position < adjacent_lenght:
                data_dictionary.setdefault(position, {}).setdefault(residue, 0)
                data_dictionary[position][residue] += 1
    number_of_peptides = len(data)
    frequency_dictionary = {}
    for position, residue_dict in data_dictionary.items():
        for residue, counts in residue_dict.items():
            frequency = counts / number_of_peptides
            frequency_dictionary.setdefault(position, {}).setdefault(residue, frequency)
    
    return frequency_dictionary
