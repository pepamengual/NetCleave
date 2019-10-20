def extract_ms_data(input_file_path):
    data = {}
    with open(input_file_path, "r") as f:
        next(f)
        next(f)
        for line in f:
            line = line.rstrip().split('","')
            peptide = line[11]
            uniprot_id = line[18].split("/")[-1]
            technique = line[79]
            qualitative_value = line[83]
            allele = line[95]
            if "mass spectrometry" in technique:
                if "Positive" in qualitative_value:
                    data.setdefault(allele, set()).add((uniprot_id, peptide))
    return data
