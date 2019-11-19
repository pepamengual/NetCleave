def probability_cleavage(data_dictionary, properties, uniprot_probabilities):
    """ Returns a dictionary of probabilities
    --> data_dictionary: a dictionary of counts of the physicochemical properties of the training data
    --> properties: a dictionary of properties for each amino acid 
    --> uniprot_probabilities: random model of uniprot. A dictionary of probabilities of the amino acids



    """
    import numpy as np
    probability_data = {}

    probability_random_physicochemical_properties = compute_random_probabilities_from_uniprot(properties, uniprot_probabilities)
    total_of_observations = sum(list(data_dictionary.values()))

    for physicochemical_properties_cleavage_region, number_of_observations in data_dictionary.items():
        list_of_physicochemical_properties = list(physicochemical_properties_cleavage_region.split("_"))
        probability_of_observation = number_of_observations/total_of_observations
        
        random_score_list = []
        for position in range(len(list_of_physicochemical_properties)):
            random_score_list.append(probability_random_physicochemical_properties[list_of_physicochemical_properties[position]])
        probability_of_random = np.prod(random_score_list)

        score = -np.log2(probability_of_observation/probability_of_random)
        probability_data.setdefault(physicochemical_properties_cleavage_region, (score, number_of_observations))
    return probability_data


def compute_random_probabilities_from_uniprot(properties, uniprot_probabilities):
    import numpy as np
    properties_amino_acid_list = {}
    for amino_acid, physicochemical_properties in properties.items():
        properties_amino_acid_list.setdefault(physicochemical_properties, []).append(uniprot_probabilities[amino_acid])
    
    probability_random_physicochemical_properties = {}
    for physicochemical_properties, probabilities_list in properties_amino_acid_list.items():
        probability_properties = np.sum(probabilities_list)
        probability_random_physicochemical_properties.setdefault(physicochemical_properties, probability_properties)
    
    return probability_random_physicochemical_properties
