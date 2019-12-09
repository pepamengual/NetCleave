from random import choices

def export_df_for_ml(large_uniprot_peptide, random_peptides, amino_acid_list, frequency_random_model_list, n, y, proteasome_ml_path, erap_ml_path):
    proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide, random_peptides, n, y)
    
    save_file(proteasome_ml_path, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, amino_acid_list, frequency_random_model_list)
    save_file(erap_ml_path, erap_cleavage_regions_ms, erap_cleavage_regions_random, amino_acid_list, frequency_random_model_list)

def get_cleavage_region(large_uniprot_peptide, random_peptides, n, y):
    x = n - y
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-n-x:-n+x] for i in range(len(large_uniprot_peptide))]
    #proteasome_cleavage_regions_random = [random_peptides[i][-n-x:-n+x] for i in range(len(random_peptides))]
    proteasome_cleavage_regions_random = [large_uniprot_peptide[i][-n-x-1:-n+x-1] for i in range(len(large_uniprot_peptide))]
    
    erap_cleavage_regions_ms = [large_uniprot_peptide[i][n-x:n+x] for i in range(len(large_uniprot_peptide))]
    #erap_cleavage_regions_random = [random_peptides[i][n-x:n+x] for i in range(len(random_peptides))]
    erap_cleavage_regions_random = [large_uniprot_peptide[i][n-x+1:n+x+1] for i in range(len(large_uniprot_peptide))]
    
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random

def make_random_peptide(length, amino_acid_list, frequency_random_model_list):
    random_peptide = "".join(choices(amino_acid_list, weights = frequency_random_model_list, k=length))
    return random_peptide

def get_random_in_data(ms_list, random_list, amino_acid_list, frequency_random_model_list):
    length = len(ms_list[0])
    ms_set = set(ms_list)
    random_set = set(random_list)
    random_in_ms_set = ms_set.intersection(random_set)
    print("There are {} unique repeated peptides".format(len(random_in_ms_set)))
    for repeated_peptide in random_in_ms_set:
        random_list.remove(repeated_peptide)
    while len(random_list) == len(ms_list):
        random_peptide = make_random_peptide(length, amino_acid_list, frequency_random_model_list)
        if not random_peptide in random_list:
            random_list.append(random_peptide)
    return ms_list, random_list

def save_file(name, ms_list, random_list, amino_acid_list, frequency_random_model_list):
    #ms_list, random_list = get_random_in_data(ms_list, random_list, amino_acid_list, frequency_random_model_list)
    header = "sequence\tclass"
    print("Saving {} file...".format(name))
    with open(name, "w") as f:
        f.write(header + "\n")
        for random_region in random_list:
            data = "{}\t{}".format(random_region, 0)
            f.write(data + "\n")
        for ms_region in ms_list:
            data = "{}\t{}".format(ms_region, 1)
            f.write(data + "\n")
