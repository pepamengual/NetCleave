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
