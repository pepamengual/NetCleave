def main_scorer(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model):
    """ Generate n random peptides
        Score n random peptides -> ERAD, PROTEASOME, BOTH
        Score n MS peptides -> ERAD, PROTEASOME, BOTH
    """
    scored_dict = {}
    print("Generating random peptides...")
    random_peptides, random_lenght_peptide, original_number_peptides, number_of_random_peptides_to_make = generate_random_peptides(large_uniprot_peptide, frequency_random_model)

    print("Scoring random peptides...")
    random_erad_best_scores, random_proteasome_best_scores, random_erad_proteasome_best_scores = score_peptides(random_peptides, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model)
    exact_number_random_erad = random_erad_best_scores[:original_number_peptides]
    exact_number_random_proteasome = random_proteasome_best_scores[:original_number_peptides]
    exact_number_random_erad_proteasome = random_erad_proteasome_best_scores[:original_number_peptides]
    print("Number of peptides in ERAD {}, PROTEASOME {}, ERAD+PROTEASOME {} from {}".format(len(exact_number_random_erad), len(exact_number_random_proteasome), len(exact_number_random_erad_proteasome), original_number_peptides))
     
    print("Scoring MS peptides...")
    ms_erad_best_scores, ms_proteasome_best_scores, ms_erad_proteasome_best_scores = score_peptides(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model)

    scored_dict.setdefault("ERAD", {}).setdefault("MS", ms_erad_best_scores)
    scored_dict.setdefault("ERAD", {}).setdefault("RANDOM", exact_number_random_erad)
    scored_dict.setdefault("PROTEASOME", {}).setdefault("MS", ms_proteasome_best_scores)
    scored_dict.setdefault("PROTEASOME", {}).setdefault("RANDOM", exact_number_random_proteasome)
    scored_dict.setdefault("ERAD+PROTEASOME", {}).setdefault("MS", ms_erad_proteasome_best_scores)
    scored_dict.setdefault("ERAD+PROTEASOME", {}).setdefault("RANDOM", exact_number_random_erad_proteasome)
    return scored_dict

def generate_random_peptides(large_uniprot_peptide, frequency_random_model):
    """ Variables
    """
    random_peptides = []
    random_lenght_peptide = 30
    original_number_peptides = len(large_uniprot_peptide)
    number_of_random_peptides_to_make = int(original_number_peptides * 2)

    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    frequency_random_model_list = list()
    for amino_acid in amino_acid_list:
        frequency = frequency_random_model[amino_acid]
        frequency_random_model_list.append(frequency)
    
    print("Making {} random peptides of lenght {}".format(number_of_random_peptides_to_make, random_lenght_peptide))
    from random import choices
    for i in range(number_of_random_peptides_to_make):
        random_peptide = "".join(choices(amino_acid_list, weights = frequency_random_model_list, k=random_lenght_peptide))
        random_peptides.append(random_peptide)
    return random_peptides, random_lenght_peptide, original_number_peptides, number_of_random_peptides_to_make

def score_peptides(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model):
    erad_best_scores = []
    proteasome_best_scores = []
    erad_proteasome_best_scores = []

    for peptide in large_uniprot_peptide:
        erad_scores = []
        proteasome_scores = []

        erad_cleavage_region_large = "".join([peptide[position] for position in erad_cleavage["large"]])
        erad_cleavage_region_short = "".join([peptide[position] for position in erad_cleavage["short"]])
        proteasome_cleavage_region_large = "".join([peptide[position] for position in proteasome_cleavage["large"]])
        proteasome_cleavage_region_short = "".join([peptide[position] for position in proteasome_cleavage["short"]])

        try:
            erad_score_large = erad_cleavage_probabilities["large"][erad_cleavage_region_large]
            erad_scores.append(erad_score_large)
        except KeyError:
            pass
        try:
            erad_score_short = erad_cleavage_probabilities["short"][erad_cleavage_region_short]
            erad_scores.append(erad_score_short)
        except KeyError:
            pass
        try:
            proteasome_score_large = proteasome_cleavage_probabilities["large"][proteasome_cleavage_region_large]
            proteasome_scores.append(proteasome_score_large)
        except KeyError:
            pass
        try:
            proteasome_score_short = proteasome_cleavage_probabilities["short"][proteasome_cleavage_region_short]
            proteasome_scores.append(proteasome_score_short)
        except KeyError:
            pass

        if len(erad_scores) >= 1 and len(proteasome_scores) >=1:
            erad_proteasome_score = min(erad_scores) + min(proteasome_scores)
            erad_proteasome_best_scores.append(erad_proteasome_score)
        if len(erad_scores) >= 1:
            min_erad_score = min(erad_scores)
            erad_best_scores.append(min_erad_score)
        if len(proteasome_scores) >= 1:
            min_proteasome_score = min(proteasome_scores)
            proteasome_best_scores.append(min_proteasome_score)

    return erad_best_scores, proteasome_best_scores, erad_proteasome_best_scores
