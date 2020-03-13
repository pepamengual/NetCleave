def export_df_for_ml(large_uniprot_peptide, proteasome_ml_path):
    proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide)
    save_file(proteasome_ml_path, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random)

def get_cleavage_region(large_uniprot_peptide):
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-12:-2] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random_1 = [large_uniprot_peptide[i][-11:-1] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random_2 = [large_uniprot_peptide[i][-10:] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random_3 = [large_uniprot_peptide[i][-13:-3] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random_4 = [large_uniprot_peptide[i][-14:-4] for i in range(len(large_uniprot_peptide))]

    proteasome_cleavage_regions_random_all = proteasome_cleavage_regions_random_1 + proteasome_cleavage_regions_random_2 + proteasome_cleavage_regions_random_3 + proteasome_cleavage_regions_random_4

    proteasome_cleavage_regions_random = list(set(proteasome_cleavage_regions_random_all) - set(proteasome_cleavage_regions_ms))
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random

def save_file(name, ms_list, random_list):
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
