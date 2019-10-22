def random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
    from random import choices
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    random_lenght_peptide = 30
    random_peptides = []
    print("Making {} random peptides of lenght {}".format(len(large_uniprot_peptide), random_lenght_peptide))
    for i in range(len(large_uniprot_peptide)):
        random_peptide = "".join(choices(amino_acid_list, k=random_lenght_peptide))
        random_peptides.append(random_peptide)
    print("Random peptides generated succesfully")
    peptides_not_found = 0
    random_scores = []
    max_value_left = max(probability_dictionary_cleavage_region_list[0]["left"].values())
    max_value_right = max(probability_dictionary_cleavage_region_list[0]["right"].values())
    max_value = max_value_left + max_value_right

    print("Max assigned")
    for peptide in random_peptides:
        score_list = []
        for pre_post_cleavage, pre_post_cleavage_name, probability_dictionary_cleavage_region in zip(pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
            random_cleavage_region_left = ""
            random_cleavage_region_right = ""
            for side, side_values in pre_post_cleavage.items():
                random_cleavage_site = "".join([peptide[r] for r in side_values])
                if side == "left":
                    random_cleavage_region_left += random_cleavage_site
                if side == "right":
                    random_cleavage_region_right += random_cleavage_site
            try:
                score_left = probability_dictionary_cleavage_region["left"][random_cleavage_region_left]
                score_right = probability_dictionary_cleavage_region["right"][random_cleavage_region_right]
                score_both = round(score_left + score_right, 6)
                score_list.append(score_both)
            except:
                score_list.append(max_value)
                peptides_not_found += 1
                continue
        if len(score_list) >= 1:
            min_score = min(score_list)
            random_scores.append(min_score)
    print("{} / {} random peptides could not be found".format(peptides_not_found / 2, len(random_peptides)))
    print("F")
    return random_scores

def scoring_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list):
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    scored_dict = {}
    print("Scoring cleavage peptides")
    for peptide in large_uniprot_peptide:
        score_list = []
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
            score_list.append(score_both)
        min_score = min(score_list)
        scored_dict.setdefault("Cleavage sites", []).append(min_score)
    print("Cleavage peptides scored succesfully")
    
    random_scores = random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list)
    scored_dict.setdefault("Random sites", random_scores)
    
    return scored_dict
