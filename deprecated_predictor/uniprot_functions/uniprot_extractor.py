def id_sequence_extractor(input_path):
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
