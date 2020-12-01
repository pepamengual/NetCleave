import random

def export_df_for_ml(amino_acid_dict_and_large_uniprot_peptide, proteasome_ml_path):
    for amino_acid, large_uniprot_peptide in amino_acid_dict_and_large_uniprot_peptide.items():
        non_cleavage_peptide = []
        for amino_acid_random, large_uniprot_random_peptide in amino_acid_dict_and_large_uniprot_peptide.items():
            if amino_acid_random != amino_acid:
                non_cleavage_peptide.extend(large_uniprot_random_peptide)
        
        proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide, non_cleavage_peptide)
        save_file(proteasome_ml_path, amino_acid, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random)

def get_cleavage_region(large_uniprot_peptide, non_cleavage_peptide):
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-12:-2] for i in range(len(large_uniprot_peptide))]
    
    
    random_1 = [large_uniprot_peptide[i][-11:-1] for i in range(len(large_uniprot_peptide))]
    random_2 = [large_uniprot_peptide[i][-10:] for i in range(len(large_uniprot_peptide))]
    random_3 = [large_uniprot_peptide[i][-13:-3] for i in range(len(large_uniprot_peptide))]
    random_4 = [large_uniprot_peptide[i][-14:-4] for i in range(len(large_uniprot_peptide))]
    
    random_5 = [non_cleavage_peptide[i][-12:-2] for i in range(len(non_cleavage_peptide))]
    random_6 = [non_cleavage_peptide[i][-11:-1] for i in range(len(non_cleavage_peptide))]
    random_7 = [non_cleavage_peptide[i][-10:] for i in range(len(non_cleavage_peptide))]
    random_8 = [non_cleavage_peptide[i][-13:-3] for i in range(len(non_cleavage_peptide))]
    random_9 = [non_cleavage_peptide[i][-14:-4] for i in range(len(non_cleavage_peptide))]

    
    proteasome_cleavage_regions_random_all = random_1 + random_2 + random_3 + random_4 + random_5 + random_6 + random_7 + random_8 + random_9

    proteasome_cleavage_regions_random = list(set(proteasome_cleavage_regions_random_all) - set(proteasome_cleavage_regions_ms))
    
    proteasome_cleavage_regions_random_choices = random.choices(proteasome_cleavage_regions_random, k=len(proteasome_cleavage_regions_ms)*20)
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random_choices

def save_file(name, amino_acid, ms_list, random_list):
    file_name = "{}_{}_sequence_class.txt".format(name, amino_acid)

    print("Saving {} file...".format(file_name))
    with open(file_name, "w") as f:
        header = "sequence\tclass"
        f.write(header + "\n")
        for random_region in random_list:
            data = "{}\t{}".format(random_region, 0)
            f.write(data + "\n")
        for ms_region in ms_list:
            data = "{}\t{}".format(ms_region, 1)
            f.write(data + "\n")
