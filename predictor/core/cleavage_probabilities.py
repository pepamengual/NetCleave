def distribution_cleavage(large_uniprot_peptide, frequency_random_model, relevant_positions):
    import numpy as np
    count = 0
    data_dictionary = {}
    for peptide in large_uniprot_peptide:
        cleavage_region_large = "".join([peptide[position] for position in relevant_positions["large"]])
        cleavage_region_short = "".join([peptide[position] for position in relevant_positions["short"]])

        data_dictionary.setdefault("large", {}).setdefault(cleavage_region_large, 0)
        data_dictionary.setdefault("short", {}).setdefault(cleavage_region_short, 0)
        data_dictionary["large"][cleavage_region_large] += 1
        data_dictionary["short"][cleavage_region_short] += 1
        count += 1
        
    frequency_dictionary = get_frequency_dictionary(data_dictionary, count, frequency_random_model, relevant_positions)
    return frequency_dictionary


def get_frequency_dictionary(data_dictionary, count, frequency_random_model, relevant_positions):
    import numpy as np
    frequency_dictionary = {}
    for side, cleavage_region_dict in data_dictionary.items():
        len_side = len(relevant_positions[side])
        for cleavage_region, counts in cleavage_region_dict.items():
            frequency = (counts / count)
            probability_cleavage_region_random = np.prod([frequency_random_model[cleavage_region[n]] for n in range(len_side)])
            probability = np.log2(frequency / probability_cleavage_region_random) * -1
            frequency_dictionary.setdefault(side, {}).setdefault(cleavage_region, probability)
    return frequency_dictionary
