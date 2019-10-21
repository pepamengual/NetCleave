def distribution_analyzer(data, adjacent_lenght):
    data_dictionary = {}
    for peptide in data:
        for position, residue in enumerate(peptide):
            if position < adjacent_lenght:
                data_dictionary.setdefault(position, {}).setdefault(residue, 0)
                data_dictionary[position][residue] += 1
    frequency_dictionary = {}
    for position, residue_dict in data_dictionary.items():
        for residue, counts in residue_dict.items():
            frequency = counts / len(data)
            frequency_dictionary.setdefault(position, {}).setdefault(residue, frequency)
    return frequency_dictionary

def distribution_plotter(frequency_dictionary_preadjacent, adjacent_lenght):
    import matplotlib.pyplot as plt

    residue_letters = "ACDEFGHIKLMNPQRSTVWY"
    print("Amino acid - position frequencies")
    for residue in residue_letters:
        frequencies_per_residue_list = [frequency_dictionary_preadjacent[i][residue] for i in range(adjacent_lenght)]
        frequencies_per_residue_str = " ".join(map(str, frequencies_per_residue_list))
        plt.plot(frequencies_per_residue_list, label=residue)
        print(residue, frequencies_per_residue_str)
    plt.legend()
    plt.show()

#def distribution_cleavage(data, adjacent_lenght, frequency_random_model, pre_pro_cleavage):
#    import numpy as np
#    count = 0
#    data_dictionary = {}
#    for peptide in data:
#        pre_cleavage = peptide[adjacent_lenght]
#        post_cleavage = peptide[adjacent_lenght + 1]
#        cleavage_region = "".join(pre_cleavage + post_cleavage)
#        data_dictionary.setdefault(cleavage_region, 0)
#        data_dictionary[cleavage_region] += 1
#        count += 1
#
#    frequency_dictionary = {}
#    for cleavage_region, counts in data_dictionary.items():
#        frequency = (counts / count)
#        probability_cleavage_region_random = (frequency_random_model[cleavage_region[0]] * frequency_random_model[cleavage_region[1]])
#        probability = np.log2(frequency / probability_cleavage_region_random) * -1
#        frequency_dictionary.setdefault(cleavage_region, probability)
#    return frequency_dictionary

def distribution_cleavage(data, frequency_random_model, pre_post_cleavage):
    import numpy as np
    count = 0
    data_dictionary = {}
    for peptide in data:
        pre_cleavage_left = peptide[pre_post_cleavage[0][0]]
        post_cleavage_left = peptide[pre_post_cleavage[0][1]]
        pre_cleavage_right = peptide[pre_post_cleavage[1][0]]
        post_cleavage_right = peptide[pre_post_cleavage[1][1]]

        cleavage_region_left = "".join(pre_cleavage_left + post_cleavage_left)
        cleavage_region_right = "".join(pre_cleavage_right + post_cleavage_right)
        
        data_dictionary.setdefault("left", {}).setdefault(cleavage_region_left, 0)
        data_dictionary.setdefault("right", {}).setdefault(cleavage_region_right, 0)
        data_dictionary["left"][cleavage_region_left] += 1
        data_dictionary["right"][cleavage_region_right] += 1
        count += 1

    frequency_dictionary = {}
    for side, cleavage_region_dict in data_dictionary.items():
        for cleavage_region, counts in cleavage_region_dict.items():
            frequency = (counts / count)
            probability_cleavage_region_random = (frequency_random_model[cleavage_region[0]] * frequency_random_model[cleavage_region[1]])
            probability = np.log2(frequency / probability_cleavage_region_random) * -1
            frequency_dictionary.setdefault(side, {}).setdefault(cleavage_region, probability)
    return frequency_dictionary
