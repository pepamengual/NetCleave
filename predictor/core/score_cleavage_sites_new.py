def random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
    from random import choices
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    random_scores = []
    for i in range(len(large_uniprot_peptide)):
        score_list = []
        for pre_post_cleavage, pre_post_cleavage_name, probability_dictionary_cleavage_region in zip(pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
            random_cleavage_region_left = ""
            random_cleavage_region_right = ""
            for side, side_values in pre_post_cleavage.items():
                len_side_values = len(side_values)
                random_cleavage_site = "".join(choices(amino_acid_list, k=len_side_values))
                if side == "left":
                    random_cleavage_region_left += random_cleavage_site
                if side == "right":
                    random_cleavage_region_right += random_cleavage_site
            random_side_list = [random_cleavage_region_left, random_cleavage_region_right]
            random_side_name = ["left", "right"]
            score_both_random = 0.0
            for side_sequence, side_name in zip(random_side_list, random_side_name):
                if side_sequence not in probability_dictionary_cleavage_region[side_name]:
                    score = max(probability_dictionary_cleavage_region[side_name].values())
                    score_both_random += score
                else:
                    score = probability_dictionary_cleavage_region[side_name][side_sequence]
                    score_both_random += score
            score_both_random = round(score_both_random, 6)
            score_list.append(score_both_random)
        min_score = min(score_list)
        random_scores.append(min_score)
    return random_scores

def scoring_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    scored_dict = {}
    for peptide in large_uniprot_peptide:
        for pre_post_cleavage, pre_post_cleavage_name, probability_dictionary_cleavage_region in zip(pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
            cleavage_region_left = ""
            cleavage_region_right = ""
            for side, side_values in pre_post_cleavage.items():
                cleavage_site = "".join([peptide[r] for r in side_values])
                if side == "left":
                    cleavage_region_left += cleavage_site
                if side == "right":
                    cleavage_region_right += cleavage_site

            score_left = probability_dictionary_cleavage_region["left"][cleavage_region_left]
            score_right = probability_dictionary_cleavage_region["right"][cleavage_region_right]
            score_both = round(score_left + score_right, 6)
        
            scored_dict.setdefault("Cleavage sites", []).append(score_both)
            
    
    random_scores = random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list)
    scored_dict.setdefault("Random sites", random_scores)
    
    return scored_dict
