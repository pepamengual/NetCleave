
def join_data(uniprot_data, uniparc_data):
    """
    Add new entries from UniParc into UniProt
    """
    print("Merging Uniprot and Uniparc data..")
    for uniprot_id, sequence in uniparc_data.items():
        if uniprot_id not in uniprot_data:
            uniprot_data.setdefault(uniprot_id, sequence)
    return uniprot_data


def locate_peptides(ms_data, uniprot_data):
    """
    Looks for adjacent C-terminal sequence for each peptide
    Remove peptides where some special character appears, that is not an standard amino acid
    Returns a list of large peptides, and a dictionary where:
        key = C-terminal residue of the peptide (IEDB)
        values = a list of peptides
    """
    print("Locating IEDB peptides into Uniprot sequences...")
    adjacent_lenght = 4
    data_dict = {}
    found_peptides = not_found_peptides = mutated_peptides = len_set = 0
    residues_set = set([residue for residue in "ACDEFGHIKLMNPQRSTVWY"])

    for uniprot_id, peptide_set in ms_data.items():
        peptide_set = set(peptide_set)
        len_set += len(peptide_set)
        if uniprot_id in uniprot_data:
            peptide_set = [peptide[-5:] for peptide in peptide_set]  # remove duplicated C-term cleavage per uniprot ID
            peptide_set = set(peptide_set)  # avoid duplicated entries of longer peptides
            for peptide in peptide_set:
                if peptide in uniprot_data[uniprot_id]:
                    found_peptides += 1
                    selected_peptide, decoy_1, decoy_2 = get_neighbour_sequence(uniprot_data, uniprot_id, peptide,
                                                                                adjacent_lenght, residues_set)
                    if selected_peptide is not None and decoy_1 is not None and decoy_2 is not None and \
                            len(selected_peptide) == 7 and len(decoy_1) == 7 and len(decoy_2) == 7:
                        data_dict.setdefault("peptides", []).append(selected_peptide)
                        data_dict.setdefault("decoys", []).append(decoy_1)
                        data_dict.setdefault("decoys", []).append(decoy_2)
                else:
                    mutated_peptides += 1
        else:
            not_found_peptides += len(peptide_set)
    
    print(f"--> Peptides to map: {len_set}")
    print(f"----> Peptides labeled with an unknown UniProt/UniParc ID: {not_found_peptides}")
    print(f"----> Peptides with known UniProt/UniParc ID that could not be mapped: {mutated_peptides}")
    print(f"----> Peptides correctly mapped: {found_peptides}")

    return data_dict


def get_neighbour_sequence(uniprot_data, uniprot_id, peptide, adjacent_lenght, residues_set):
    adjacent_sequences = uniprot_data[uniprot_id].split(peptide)
    post_sequence = adjacent_sequences[1][:adjacent_lenght+1]
    if len(post_sequence) == adjacent_lenght+1:
        large_peptide = "".join(peptide + post_sequence)
        large_peptide_set = set(large_peptide)
        if not large_peptide_set - residues_set:
            selected_peptide = large_peptide[1:8]
            decoy_1 = large_peptide[0:7]
            decoy_2 = large_peptide[2:9]
            return selected_peptide, decoy_1, decoy_2
        else:
            return None, None, None
    else:
        return None, None, None
