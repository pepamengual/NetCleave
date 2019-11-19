from random import choices

def generate_random_peptides(large_uniprot_peptide, frequency_random_model):
    random_peptides = []
    random_lenght_peptide = 30
    original_number_peptides = len(large_uniprot_peptide)

    amino_acid_list = list("ACDEFGHIKLMNPQRSTVWY")
    frequency_random_model_list = list()

    for amino_acid in amino_acid_list:
        frequency = frequency_random_model[amino_acid]
        frequency_random_model_list.append(frequency)

    print("Making {} random peptides of lenght {}".format(original_number_peptides, random_lenght_peptide))
    
    for i in range(original_number_peptides):
        random_peptide = "".join(choices(amino_acid_list, weights = frequency_random_model_list, k=random_lenght_peptide))
        random_peptides.append(random_peptide)
    return random_peptides
