import pandas as pd


def extract_uniparc_data(uniparc_path_headers, uniparc_path_sequence):
    print("Extracting protein sequence data from Uniparc...")
    header_dict = get_headers_uniprot_uniparc(uniparc_path_headers)
    data = {}
    with open(uniparc_path_sequence, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                uniparc_id = line.split(" ")[0][1:] # remove > symbol
                uniprot_id = header_dict[uniparc_id]
                data.setdefault(uniprot_id, "")
            else:
                data[uniprot_id] += line
    return data


def get_headers_uniprot_uniparc(uniparc_path_headers):
    df = pd.read_csv(uniparc_path_headers, sep="\t")
    data = pd.Series(df["yourlist:M20200416A94466D2655679D1FD8953E075198DA854EB3ES"].values, index=df.Entry).to_dict()
    return data
