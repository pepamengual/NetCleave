def score_peptides(peptide_list, relevant_positions, properties, probability_data):
    import numpy as np
    peptide_score_list = []
    for peptide in peptide_list:
        cleavage_region = "".join([peptide[position] for position in relevant_positions])

        properties_all_residues = []
        for i, amino_acid in enumerate(cleavage_region):
            properties_of_amino_acid = properties[amino_acid]
            properties_all_residues.append(properties_of_amino_acid)
        properties_all_residues = "_".join(properties_all_residues)
    
        try:
            probability_of_cleavage = probability_data[properties_all_residues][0]
            peptide_score_list.append(probability_of_cleavage)
        except KeyError:
            #random_number = np.random.normal(loc=2, scale=1)
            #peptide_score_list.append(random_number)
            pass
    return peptide_score_list
