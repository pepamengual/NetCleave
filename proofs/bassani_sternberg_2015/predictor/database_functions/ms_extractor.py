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
            mhc_class = line[98]
            if "mass spectrometry" in technique and "Positive" in qualitative_value and mhc_class == "I" and len(uniprot_id) > 1:
                data.setdefault(uniprot_id, set()).add(peptide)
    return data
