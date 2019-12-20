def compute_cleavage_regions(large_uniprot_peptide, random_peptides, n):
    proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random = get_cleavage_region(large_uniprot_peptide, random_peptides, n)
    

    name = "proteasome"
    with open("data/ml_dataframes/{}_positive_data.txt".format(name), "w") as f:
        for peptide in proteasome_cleavage_regions_ms:
            f.write(peptide + "\n")
    save_file(name, proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random)
    name = "erap"
    with open("data/ml_dataframes/{}_positive_data.txt".format(name), "w") as f:
        for peptide in erap_cleavage_regions_ms:
            f.write(peptide + "\n")
    save_file(name, erap_cleavage_regions_ms, erap_cleavage_regions_random)

def get_cleavage_region(large_uniprot_peptide, random_peptides, n):
    y = n - 1
    proteasome_cleavage_regions_ms = [large_uniprot_peptide[i][-n-y:-n+y] for i in range(len(large_uniprot_peptide))]
    proteasome_cleavage_regions_random = [random_peptides[i][-n-y:-n+y] for i in range(len(random_peptides))]
    erap_cleavage_regions_ms = [large_uniprot_peptide[i][n-y:n+y] for i in range(len(large_uniprot_peptide))]
    erap_cleavage_regions_random = [random_peptides[i][n-y:n+y] for i in range(len(random_peptides))]
    return proteasome_cleavage_regions_ms, proteasome_cleavage_regions_random, erap_cleavage_regions_ms, erap_cleavage_regions_random

def save_file(name, ms_list, random_list):
    features_all = {"A": [0.62, 27.5, 8.1, 0.046, 1.181, 0.007187, -0.5, 2, 71.0788], "C": [0.29, 44.6, 5.5, 0.128, 1.461, -0.03661, -1, 2, 103.1388], "D": [-0.9, 40, 13, 0.105, 1.587, -0.02382, 3, 4, 115.0886], "E": [-0.74, 62, 12.3, 0.151, 1.862, 0.006802, 3, 4, 129.1155], "F": [1.19, 115.5, 5.2, 0.29, 2.228, 0.037552, -2.5, 2, 147.1766], "G": [0.48, 0, 9, 0, 0.881, 0.179052, 0, 2, 57.0519], "H": [-0.4, 79, 10.4, 0.23, 2.025, -0.01069, -0.5, 4, 137.1411], "I": [1.38, 93.5, 5.2, 0.186, 1.81, 0.021631, -1.8, 2, 113.1594], "K": [-1.5, 100, 11.3, 0.219, 2.258, 0.017708, 3, 2, 128.1741], "L": [1.06, 93.5, 4.9, 0.186, 1.931, 0.051672, -1.8, 2, 113.1594], "M": [0.64, 94.1, 5.7, 0.221, 2.034, 0.002683, -1.3, 2, 131.1986], "N": [-0.78, 58.7, 11.6, 0.134, 1.655, 0.005392, 2, 4, 114.1039], "P": [0.12, 41.9, 8, 0.131, 1.468, 0.239531, 0, 2, 97.1167], "Q": [-0.85, 80.7, 10.5, 0.18, 1.932, 0.049211, 0.2, 4, 128.1307], "R": [-2.53, 105, 10.5, 0.291, 2.56, 0.043587, 3, 4, 156.1875], "S": [-0.18, 29.3, 9.2, 0.062, 1.298, 0.004627, 0.3, 4, 87.0782], "T": [-0.05, 51.3, 8.6, 0.108, 1.525, 0.003352, -0.4, 4, 101.1051], "V": [1.08, 71.5, 5.9, 0.14, 1.645, 0.057004, -1.5, 2, 99.1326], "W": [0.81, 145.5, 5.4, 0.409, 2.663, 0.037977, -3.4, 3, 186.2132], "Y": [0.26, 117.3, 6.2, 0.298, 2.368, 0.023599, -2.3, 3, 163.176]}
    #ref : https://docs.google.com/spreadsheets/d/1T0do5Jt4Mwb8xs1JfEZj-8oqw-VLtqW1kRW3AQuQPoQ/edit?usp=sharing

    len_peptide = len(ms_list[0])
    len_features = len(features_all["A"])
    len_total = len_peptide * len_features
    header_list = list(range(len_total))
    header_list.append("class")
    header = "\t".join(str(i) for i in header_list)
    
    output_path = "data/ml_dataframes/{}_dataframe_for_ml.txt".format(name)
    print("Saving {} file...".format(output_path))
    with open(output_path, "w") as f:
        header = "\t".join(str(i) for i in header_list)
        f.write(header + "\n")
        for random_region in random_list:
            value_list = []
            for residue in random_region:
                for feature in features_all[residue]:
                    value_list.append(str(feature))
            feature_data = "\t".join(value_list)
            data = "{}\t{}".format(feature_data, 0)
            f.write(data + "\n")
        for ms_region in ms_list:
            value_list = []
            for residue in ms_region:
                for feature in features_all[residue]:
                    value_list.append(str(feature))
            feature_data = "\t".join(value_list)
            data = "{}\t{}".format(feature_data, 1)
            f.write(data + "\n")
