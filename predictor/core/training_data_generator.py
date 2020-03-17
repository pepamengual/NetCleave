import random
from pathlib import Path

def prepare_cleavage_data(proteasome_dictionary, export_path):
    """ Iterates for every key of proteasome_dictionary: C-terminal residue of MS peptides
        Creates a list consisting of all other peptides with different C-terminal residue
        Gets cleavage regions (+3) residues from the C-terminal samples
        Generates two equal size lists of cleavaged and non cleavaged samples having the same residue in C-terminal
        Exports data in a given path
    """
    for amino_acid, proteasome_peptides in sorted(proteasome_dictionary.items()):
        print("--> Preparing training data for {}...".format(amino_acid))
        all_other_peptides = []
        for amino_acid_, uncleaved_list_of_peptides in proteasome_dictionary.items():
            if amino_acid_ != amino_acid:
                all_other_peptides.extend(uncleaved_list_of_peptides)
        proteasome_cleavages, equal_size_non_cleavaged_list = get_cleavage_region(amino_acid, proteasome_peptides, all_other_peptides)
        if len(proteasome_cleavages) != len(equal_size_non_cleavaged_list):
            print("WARNING: cleavage sample size ({}) differs from non cleavaged ones ({})".format(len(proteasome_cleavages), len(equal_size_non_cleavaged_list)))
        export_data(export_path, amino_acid, proteasome_cleavages, equal_size_non_cleavaged_list)

def get_cleavage_region(amino_acid, proteasome_peptides, all_other_peptides):
    proteasome_cleavages = [proteasome_peptides[i][-8:-1] for i in range(len(proteasome_peptides))]
    random_cleavage_1 = [proteasome_peptides[i][-9:-2] for i in range(len(proteasome_peptides))]
    random_cleavage_2 = [proteasome_peptides[i][-7:] for i in range(len(proteasome_peptides))]
    random_cleavage_3 = [all_other_peptides[i][-8:-1] for i in range(len(all_other_peptides))]
    random_cleavage_4 = [all_other_peptides[i][-9:-2] for i in range(len(all_other_peptides))]
    random_cleavage_5 = [all_other_peptides[i][-7:] for i in range(len(all_other_peptides))]
    random_list = random_cleavage_1 + random_cleavage_2 + random_cleavage_3 + random_cleavage_4 + random_cleavage_5
    
    non_cleavaged_samples = []
    for random_peptide in random_list:
        proteasome_acting_amino_acid = random_peptide[3]
        if proteasome_acting_amino_acid == amino_acid:
            non_cleavaged_samples.append(random_peptide)
    putative_non_cleavaged_samples = list(set(non_cleavaged_samples) - set(proteasome_cleavages)) #sanity filter check
    equal_size_non_cleavaged_list = random.choices(putative_non_cleavaged_samples, k=len(proteasome_cleavages)) #same amount of negative entries, to check
    
    return proteasome_cleavages, equal_size_non_cleavaged_list

def export_data(export_path, amino_acid, proteasome_cleavages, random_cleavages):
    Path(export_path).mkdir(parents=True, exist_ok=True)
    file_name = "{}/{}_{}_sequence_class.txt".format(export_path, export_path.split("/")[-1], amino_acid)
    with open(file_name, "w") as f:
        header = "sequence\tclass\n"
        f.write(header)
        for random_region in random_cleavages:
            data = "{}\t{}\n".format(random_region, 0)
            f.write(data)

        for cleavage_region in proteasome_cleavages:
            data = "{}\t{}\n".format(cleavage_region, 1)
            f.write(data)
