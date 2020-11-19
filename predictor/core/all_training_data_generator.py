import random
from pathlib import Path
#from predictor.core.generate_peptidome import generate_peptidome

def prepare_cleavage_data(selected_dictionary, export_path):
    """ Iterates for every key of selected_dictionary: C-terminal residue of MS peptides
        Creates a list consisting of all other peptides with different C-terminal residue
        Gets cleavage regions (+3) residues from the C-terminal samples
        Generates two equal size lists of cleavaged and non cleavaged samples having the same residue in C-terminal
        Exports data in a given path
    """

    Path(export_path).mkdir(parents=True, exist_ok=True)
    file_name = "{}/{}_sequence_class.txt".format(export_path, export_path.split("/")[-1])

    selected_cleavages = selected_dictionary["peptides"]
    random_cleavages = selected_dictionary["decoys"]

    with open(file_name, "w") as f:
        header = "sequence\tclass\n"
        f.write(header)
        for random_region in random_cleavages:
            data = "{}\t{}\n".format(random_region, 0)
            f.write(data)

        for cleavage_region in selected_cleavages:
            data = "{}\t{}\n".format(cleavage_region, 1)
            f.write(data)

