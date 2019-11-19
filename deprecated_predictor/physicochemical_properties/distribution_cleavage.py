def distribution_cleavage(large_uniprot_peptide, relevant_positions, properties):
    """ Returns a dictionary of counts
        --> large_uniprot_peptide: list of peptides
        --> relevant_positions: list of positions to evaluate within the cleavage region of the peptide
        --> properties: a dictionary of properties for each amino acid 

        Iterates over all training peptides
        Loads a dictionary of physicochemical properties of each amino acid
        Gets the cleavage region of the peptide
        Gets the physicochemical properties of each amino acid of the cleavage region
        Joins in a string the physicochemical properties
        Counts how many times this physicochemical properties appear on the training dataset
    """
    data_dictionary = {}
    for peptide in large_uniprot_peptide:
        cleavage_region = "".join([peptide[position] for position in relevant_positions])
        
        properties_all_residues = []
        for i, amino_acid in enumerate(cleavage_region):
            properties_of_amino_acid = properties[amino_acid]
            properties_all_residues.append(properties_of_amino_acid)
        properties_all_residues = "_".join(properties_all_residues)
        
        data_dictionary.setdefault(properties_all_residues, 0)
        data_dictionary[properties_all_residues] += 1
    
    return data_dictionary
