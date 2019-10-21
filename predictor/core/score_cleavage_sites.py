def scoring_peptides(large_uniprot_peptide, random_position_list, probability_dictionary_cleavage_region):
    scored_dict = {}
    for random_position in random_position_list:
        random_pre_post_cleavage = [[random_position - 1, random_position],[(random_position + 1) * -1, random_position * -1]]
        for peptide in large_uniprot_peptide:
            pre_cleavage_left = peptide[random_pre_post_cleavage[0][0]]
            post_cleavage_left = peptide[random_pre_post_cleavage[0][1]]
            pre_cleavage_right = peptide[random_pre_post_cleavage[1][0]]
            post_cleavage_right = peptide[random_pre_post_cleavage[1][1]]
            
            cleavage_region_left = "".join(pre_cleavage_left + post_cleavage_left)
            cleavage_region_right = "".join(pre_cleavage_right + post_cleavage_right)

            score_left = probability_dictionary_cleavage_region["left"][cleavage_region_left]
            score_right = probability_dictionary_cleavage_region["right"][cleavage_region_right]
            score_both = round(score_left + score_right, 6)
            scored_dict.setdefault(random_position, []).append(score_both)
    
    from random import choices
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    for peptide in large_uniprot_peptide:
        random_choice_1 = "".join(choices(amino_acid_list, k=2))
        random_choice_2 = "".join(choices(amino_acid_list, k=2))
        score_left = probability_dictionary_cleavage_region["left"][random_choice_1]
        score_right = probability_dictionary_cleavage_region["right"][random_choice_2]
        score_both = round(score_left + score_right, 6)
        scored_dict.setdefault("random", []).append(score_both)

    return scored_dict
