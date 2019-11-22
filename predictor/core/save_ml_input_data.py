def export_df_for_ml(large_uniprot_peptide, random_peptides, n, proteasome_ml_path, erap_ml_path):
    proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide, random_peptides, n)
    
    save_file(proteasome_ml_path, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random)
    save_file(erap_ml_path, erap_cleavage_regions_ms, erap_cleavage_regions_random)

def get_cleavage_region(large_uniprot_peptide, random_peptides, n):
    y = n - 1
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-n-y:-n+y] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random = [random_peptides[i][-n-y:-n+y] for i in range(len(random_peptides))]
    erap_cleavage_regions_ms = [large_uniprot_peptide[i][n-y:n+y] for i in range(len(large_uniprot_peptide))]
    erap_cleavage_regions_random = [random_peptides[i][n-y:n+y] for i in range(len(random_peptides))]
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random

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
