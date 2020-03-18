import pandas as pd

def extract_peptide_data(input_file_path, mhc_class_type):
    """
    Extracts mass spectrometry positive binding peptides of all MHC-I alleles in IEDB
    Returns a dictionary:
        key = uniprot_id
        values = a set of binding peptides
    """
    print("Extracting peptide data from IEDB...")
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
            if "mass spectrometry" in technique and "Positive" in qualitative_value and mhc_class == mhc_class_type and len(uniprot_id) > 1:
                data.setdefault(uniprot_id, set()).add(peptide)
    return data

def extract_peptide_data_pandas(input_file_path, conditions_dictionary):
    print("Extracting peptide data from IEDB...")
    df = pd.read_csv(input_file_path, header=1, usecols=list(conditions_dictionary.keys()))
    df = df.dropna()
    df = df.reset_index(drop=True)
    for condition_type, condition_values in conditions_dictionary.items():
        if condition_values is not None:
            condition_search, condition_value = condition_values[0], condition_values[1]
            if condition_search == "contains":
                df = df[df[condition_type].str.contains(condition_value, regex=False)]
            if condition_search == "match":
                df = df[df[condition_type] == condition_value]
    exporting_df = df[["Description", "Parent Protein IRI"]]
    exporting_df = exporting_df.drop_duplicates(keep="first")
    exporting_df["Parent Protein IRI"] = exporting_df["Parent Protein IRI"].str.split("/").str[-1]
    data = exporting_df.groupby("Parent Protein IRI")["Description"].apply(list).to_dict()
    return data
