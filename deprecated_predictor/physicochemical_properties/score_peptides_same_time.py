def score_peptides(peptide_list, relevant_positions, relevant_positions_2, properties, probability_data, probability_data_2):
    import numpy as np
    peptide_score_list = []
    for peptide in peptide_list:
        f_score = []
        cleavage_region = "".join([peptide[position] for position in relevant_positions])
        cleavage_region_2 = "".join([peptide[position] for position in relevant_positions_2])
        properties_all_residues = []
        
        for i, amino_acid in enumerate(cleavage_region):
            properties_of_amino_acid = properties[amino_acid]
            properties_all_residues.append(properties_of_amino_acid)
        properties_all_residues = "_".join(properties_all_residues)
        try:
            probability_of_cleavage = probability_data[properties_all_residues][0]
            f_score.append(probability_of_cleavage)
        except KeyError:
            pass

        properties_all_residues_2 = []
        for i, amino_acid in enumerate(cleavage_region_2):
            properties_of_amino_acid = properties[amino_acid]
            properties_all_residues_2.append(properties_of_amino_acid)
        properties_all_residues_2 = "_".join(properties_all_residues_2)
        try:
            probability_of_cleavage = probability_data_2[properties_all_residues_2][0]
            f_score.append(probability_of_cleavage)
        except KeyError:
            pass
        
        if len(f_score) == 2:
            peptide_score_list.append(np.sum(f_score))
    return peptide_score_list
