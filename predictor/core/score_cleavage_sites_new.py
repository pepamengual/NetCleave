def random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list, frequency_random_model):
    from random import choices
    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    frequency_random_model_list = list()
    for amino_acid in amino_acid_list:
        frequency = frequency_random_model[amino_acid]
        frequency_random_model_list.append(frequency)
    
    random_lenght_peptide = 30
    random_peptides = []
    original_number_peptides = len(large_uniprot_peptide)
    number_of_random_peptides = int(original_number_peptides + original_number_peptides/5)
    print("Making {} random peptides of lenght {}".format(number_of_random_peptides, random_lenght_peptide))
    for i in range(number_of_random_peptides):
        random_peptide = "".join(choices(amino_acid_list, weights = frequency_random_model_list, k=random_lenght_peptide))
        random_peptides.append(random_peptide)
    print("Random peptides generated succesfully")
    random_scores = []
    random_scored_peptides_succesfully = 0
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
                continue
        if len(score_list) >= 1:
            min_score = min(score_list)
            random_scores.append(min_score)
            random_scored_peptides_succesfully += 1

    exact_name_random_peptides = random_scores[:original_number_peptides]
    print("{} / {} random peptides scored, using {}".format(random_scored_peptides_succesfully, number_of_random_peptides, len(exact_name_random_peptides)))
    return exact_name_random_peptides

def scoring_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list, frequency_random_model):
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
    
    random_scores = random_peptides(large_uniprot_peptide, pre_post_cleavage_list, pre_post_cleavage_names, probability_dictionary_cleavage_region_list, frequency_random_model)
    scored_dict.setdefault("Random sites", random_scores)
    
    return scored_dict
