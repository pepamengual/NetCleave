def main_scorer(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model):
    """ Generate n random peptides
        Score n random peptides -> ERAD, PROTEASOME, BOTH
        Score n MS peptides -> ERAD, PROTEASOME, BOTH
    """
    scored_dict = {}
    print("Generating random peptides...")
    random_peptides, random_lenght_peptide, original_number_peptides, number_of_random_peptides_to_make = generate_random_peptides(large_uniprot_peptide, frequency_random_model)

    print("Scoring MS peptides...")
    ms_erad_best_scores, ms_proteasome_best_scores, ms_erad_proteasome_best_scores = score_peptides(large_uniprot_peptide, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model)

    print("Scoring random peptides...")
    random_erad_best_scores, random_proteasome_best_scores, random_erad_proteasome_best_scores = score_peptides(random_peptides, erad_cleavage, proteasome_cleavage, erad_cleavage_probabilities, proteasome_cleavage_probabilities, frequency_random_model)

    exact_number_random_erad = random_erad_best_scores[:len(ms_erad_best_scores)]
    exact_number_random_proteasome = random_proteasome_best_scores[:len(ms_proteasome_best_scores)]
    exact_number_random_erad_proteasome = random_erad_proteasome_best_scores[:len(ms_erad_proteasome_best_scores)]
    
    print("Number of peptides in ERAD {} {}, PROTEASOME {} {}, ERAD+PROTEASOME {} {}".format(len(exact_number_random_erad), len(ms_erad_best_scores), len(exact_number_random_proteasome), len(ms_proteasome_best_scores), len(exact_number_random_erad_proteasome), len(ms_erad_proteasome_best_scores)))

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
    number_of_random_peptides_to_make = int(original_number_peptides)

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

    cleavage_list_name = []
    cleavage_names = []

    cleavage_types = list(erad_cleavage.keys()) # large, short
    possible_cleavages = len(cleavage_types) # 2
    if possible_cleavages == 1:
        cleavage_list_name.extend(cleavage_types*2)
    else:
        cleavage_list_name.extend(cleavage_types*possible_cleavages)

    cleavage_methods = ["erad", "proteasome"]
    for method in cleavage_methods:
        cleavage_names.extend([method]*possible_cleavages)
    
    cleavage_probabilities = []
    cleavage_probab = [erad_cleavage_probabilities, proteasome_cleavage_probabilities]
    cleavage_probabilities.extend(cleavage_probab*possible_cleavages)
    cleavage_sequences = [erad_cleavage, proteasome_cleavage]

    for peptide in large_uniprot_peptide:
        data_scores = {"erad": [], "proteasome": []}
        cleavage_list = []
        for cleavage_sequence in cleavage_sequences:
            for cleavage_name in cleavage_types:
                sequence = "".join([peptide[position] for position in cleavage_sequence[cleavage_name]])
                cleavage_list.append(sequence)
        
        for sequence, cleavage_name, cleavage_region, cleavage_probability in zip(cleavage_list, cleavage_list_name, cleavage_names, cleavage_probabilities):
            try:
                score = cleavage_probability[cleavage_name][sequence]
                data_scores[cleavage_region].append(score)
            except KeyError:
                pass
        
        if len(data_scores["erad"]) >= 1 and len(data_scores["proteasome"]) >= 1:
            erad_proteasome_score = min(data_scores["erad"]) + min(data_scores["proteasome"])
            erad_proteasome_best_scores.append(erad_proteasome_score)
        if len(data_scores["erad"]) >= 1:
            min_erad_score = min(data_scores["erad"])
            erad_best_scores.append(min_erad_score)
        if len(data_scores["proteasome"]) >= 1:
            min_proteasome_score = min(data_scores["proteasome"]) #CHANGED MIN TO MAX TO SEE HOW THE PERFORMANCE CHANGES
            proteasome_best_scores.append(min_proteasome_score)

    return erad_best_scores, proteasome_best_scores, erad_proteasome_best_scores
