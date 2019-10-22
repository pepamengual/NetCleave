def scoring_peptides(large_uniprot_peptide, probability_dictionary_cleavage_region, pre_post_cleavage):
    from random import choices
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    scored_dict = {}
    for peptide in large_uniprot_peptide:
        cleavage_region_left = ""
        cleavage_region_right = ""

        random_cleavage_region_left = ""
        random_cleavage_region_right = ""
        for side, side_values in pre_post_cleavage.items():
            len_side_values = len(side_values)
            random_cleavage_site = "".join(choices(amino_acid_list, k=len_side_values))
            cleavage_site = "".join([peptide[r] for r in side_values])
            if side == "left":
                cleavage_region_left += cleavage_site
                random_cleavage_region_left += random_cleavage_site
            if side == "right":
                cleavage_region_right += cleavage_site
                random_cleavage_region_right += random_cleavage_site

        score_left = probability_dictionary_cleavage_region["left"][cleavage_region_left]
        score_right = probability_dictionary_cleavage_region["right"][cleavage_region_right]
        score_both = round(score_left + score_right, 6)
        
        if random_cleavage_region_left not in probability_dictionary_cleavage_region["left"]:
            score_left_random = max(probability_dictionary_cleavage_region["left"].values())
        else:
            score_left_random = probability_dictionary_cleavage_region["left"][random_cleavage_region_left]
        
        if random_cleavage_region_right not in probability_dictionary_cleavage_region["right"]:
            score_right_random = max(probability_dictionary_cleavage_region["right"].values())
        else:
            score_right_random = probability_dictionary_cleavage_region["right"][random_cleavage_region_right]
        
        score_both_random = round(score_left_random + score_right_random, 6)
        
        scored_dict.setdefault("Cleavage sites", []).append(score_both)
        scored_dict.setdefault("Random sites", []).append(score_both_random)
    
    return scored_dict
