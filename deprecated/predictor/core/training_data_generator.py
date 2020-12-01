import random
from pathlib import Path
from predictor.core.generate_peptidome import generate_peptidome

def prepare_cleavage_data(selected_dictionary, export_path, sequence_data):
    """ Iterates for every key of selected_dictionary: C-terminal residue of MS peptides
        Creates a list consisting of all other peptides with different C-terminal residue
        Gets cleavage regions (+3) residues from the C-terminal samples
        Generates two equal size lists of cleavaged and non cleavaged samples having the same residue in C-terminal
        Exports data in a given path
    """
    peptidome = generate_peptidome(sequence_data)
    peptidome_residues = split_peptidome(peptidome)

    for amino_acid, selected_peptides in sorted(selected_dictionary.items()):
        print("--> Preparing training data for {}...".format(amino_acid))
        selected_cleavages, decoy_cleavages = get_cleavage_region(amino_acid, selected_peptides, peptidome_residues[amino_acid])
        export_data(export_path, amino_acid, selected_cleavages, decoy_cleavages)

def split_peptidome(peptidome):
    peptidome_residues = {}
    residues = "ACDEFGHIKLMNPQRSTVWY"
    residues_set = set([residue for residue in residues])
    
    for peptide in peptidome:
        peptide_set = set(peptide)
        if not peptide_set - residues_set:
            peptidome_residues.setdefault(peptide[3], []).append(peptide)
    return peptidome_residues
        

def get_cleavage_region(amino_acid, selected_peptides, decoy_cleavages):
    residues = "ACDEFGHIKLMNPQRSTVWY"
    residues_set = set([residue for residue in residues])
    selected_cleavages_no_filtered = [selected_peptides[i][-8:-1] for i in range(len(selected_peptides))]

    selected_cleavages = []
    for peptide in selected_cleavages_no_filtered:
        peptide_set = set(peptide)
        if not peptide_set - residues_set:
            selected_cleavages.append(peptide)
    
    decoy_cleavages_ = list(set(decoy_cleavages) - set(selected_cleavages))
    decoy_cleavages_ = random.choices(decoy_cleavages_, k=len(selected_cleavages))
    
    print("{} amino acid has {} cleavage sites and {} decoys from {} generated decoys".format(amino_acid, len(selected_cleavages), len(decoy_cleavages_), len(decoy_cleavages)))
    
    return selected_cleavages, decoy_cleavages_

def export_data(export_path, amino_acid, selected_cleavages, random_cleavages):
    Path(export_path).mkdir(parents=True, exist_ok=True)
    file_name = "{}/{}_{}_sequence_class.txt".format(export_path, export_path.split("/")[-1], amino_acid)
    with open(file_name, "w") as f:
        header = "sequence\tclass\n"
        f.write(header)
        for random_region in random_cleavages:
            data = "{}\t{}\n".format(random_region, 0)
            f.write(data)

        for cleavage_region in selected_cleavages:
            data = "{}\t{}\n".format(cleavage_region, 1)
            f.write(data)
