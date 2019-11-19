def compute_cleavage_regions(large_uniprot_peptide, random_peptides, n):
    proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide, random_peptides, n)

    name = "proteasome"
    save_file(name, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random)
    name = "erap"
    save_file(name, erap_cleavage_regions_ms, erap_cleavage_regions_random)

def get_cleavage_region(large_uniprot_peptide, random_peptides, n):
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-n-3:-n+3] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random = [random_peptides[i][-n-3:-n+3] for i in range(len(random_peptides))]
    erap_cleavage_regions_ms = [large_uniprot_peptide[i][n-3:n+3] for i in range(len(large_uniprot_peptide))]
    erap_cleavage_regions_random = [random_peptides[i][n-3:n+3] for i in range(len(random_peptides))]
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random

def save_file(name, ms_list, random_list):
    output_path = "data/ml_dataframes/{}_dataframe_for_ml.txt".format(name)
    print("Saving {} file...".format(output_path))
    with open(output_path, "w") as f:
        header = "cleavage_region\tclass"
        f.write(header + "\n")
        for random_region in random_list:
            data = "{}\t{}".format(random_region, 0)
            f.write(data + "\n")
        for ms_region in ms_list:
            data = "{}\t{}".format(ms_region, 1)
            f.write(data + "\n")
