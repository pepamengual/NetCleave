
def extract_uniprot_data(input_path):
    """
    Reads uniprot file and converts it into a dictionary where:
        key = uniprot_id
        value = a string of the FASTA sequence
    """
    print("Extracting protein sequence data from Uniprot...")
    data = {}
    with open(input_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                uniprot_id = line.split("|")[1]
                data.setdefault(uniprot_id, "")
            else:
                data[uniprot_id] += line
    return data
