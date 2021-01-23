def join_data(uniprot_data, uniparc_data):
    print("Merging Uniprot and Uniparc data..")
    for uniprot_id, sequence in uniparc_data.items():
        if not uniprot_id in uniprot_data:
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
    residues = "ACDEFGHIKLMNPQRSTVWY"
    residues_set = set([residue for residue in residues])

    for uniprot_id, peptide_set in ms_data.items():
        peptide_set = set(peptide_set)
        len_set += len(peptide_set)
        if uniprot_id in uniprot_data:
            peptide_set = [peptide[-5:] for peptide in peptide_set] #remove duplicated C-term cleavage per uniprot ID
            peptide_set = set(peptide_set) #avoid duplicated entries of longer peptides
            for peptide in peptide_set:
                if peptide in uniprot_data[uniprot_id]:
                    found_peptides += 1
                    selected_peptide, decoy_1, decoy_2 = get_neighbour_sequence(uniprot_data, uniprot_id, peptide, adjacent_lenght, residues_set)
                    if selected_peptide != None and decoy_1 != None and decoy_2 != None and len(selected_peptide) == 7 and len(decoy_1) == 7 and len(decoy_2) == 7:
                        data_dict.setdefault("peptides", []).append(selected_peptide)
                        data_dict.setdefault("decoys", []).append(decoy_1)
                        data_dict.setdefault("decoys", []).append(decoy_2)
                else:
                    mutated_peptides += 1
        else:
            not_found_peptides += len(peptide_set)
    
    print("{} unique peptides".format(len_set))
    print("{}/{} peptides have been found/not found in Uniprot/Uniparc".format(found_peptides, not_found_peptides))
    print("{} mutation peptides".format(mutated_peptides))
    return data_dict

def get_neighbour_sequence(uniprot_data, uniprot_id, peptide, adjacent_lenght, residues_set):
    adjacent_sequences = uniprot_data[uniprot_id].split(peptide)
    post_sequence = adjacent_sequences[1][:adjacent_lenght+1]
    if len(post_sequence) == adjacent_lenght+1:
        large_peptide = "".join(peptide + post_sequence)
        large_peptide_set = set(large_peptide)
        if not large_peptide_set - residues_set:
            selected_peptide = large_peptide[1:8]#[-8:-1]
            decoy_1 = large_peptide[0:7]#[-7:]
            decoy_2 = large_peptide[2:9]#[-9:-2]
            return selected_peptide, decoy_1, decoy_2
        else:
            return None, None, None
    else:
        return None, None, None

