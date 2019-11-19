def generate_random_peptides(large_uniprot_peptide, uniprot_probabilities):
    """ Variables
    """
    random_peptides = []
    random_lenght_peptide = 30
    original_number_peptides = len(large_uniprot_peptide)
    number_of_random_peptides_to_make = int(original_number_peptides)

    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    uniprot_probabilities_list = list()
    for amino_acid in amino_acid_list:
        frequency = uniprot_probabilities[amino_acid]
        uniprot_probabilities_list.append(frequency)

    print("Making {} random peptides of lenght {}".format(number_of_random_peptides_to_make, random_lenght_peptide))
    from random import choices
    for i in range(number_of_random_peptides_to_make):
        random_peptide = "".join(choices(amino_acid_list, weights = uniprot_probabilities_list, k=random_lenght_peptide))
        random_peptides.append(random_peptide)
    return random_peptides

