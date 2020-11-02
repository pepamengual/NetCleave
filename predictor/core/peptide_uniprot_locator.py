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
    amino_acids_possible_set = set_creator_of_amino_acids("ACDEFGHIKLMNPQRSTVWY")
    adjacent_lenght = 4
    data_dict, mutation_dict = {}, {}
    found_peptides = not_found_peptides = mutated_peptides = len_set = 0
    for uniprot_id, peptide_set in ms_data.items():
        peptide_set = set(peptide_set)
        len_set += len(peptide_set)
        if uniprot_id in uniprot_data:
            peptide_set = [peptide[-5:] for peptide in peptide_set] #remove duplicated C-term cleavage per uniprot ID
            peptide_set = set(peptide_set) #avoid duplicated entries of longer peptides
            for peptide in peptide_set:
                if peptide in uniprot_data[uniprot_id]:
                    found_peptides += 1
                    last_peptide_residue, large_peptide = get_neighbour_sequence(uniprot_data, uniprot_id, peptide, adjacent_lenght, amino_acids_possible_set, 0)
                    if last_peptide_residue != None and large_peptide != None:
                        data_dict.setdefault(last_peptide_residue, []).append(large_peptide)
                else:
                    mutated_peptides += 1
        else:
            not_found_peptides += len(peptide_set)
               # else:
               #     last_peptide_residue, large_peptide = find_single_mutations_in_c_terminal(peptide, uniprot_data, uniprot_id, adjacent_lenght, amino_acids_possible_set)
               #     if last_peptide_residue != None and large_peptide != None:
               #         mutation_dict.setdefault(uniprot_id, []).append((peptide, large_peptide))
    print("{} unique peptides".format(len_set))
    print("{}/{} peptides have been found/not found in Uniprot/Uniparc".format(found_peptides, not_found_peptides))
    print("{} mutation peptides".format(mutated_peptides))
    return data_dict, mutation_dict

def find_single_mutations_in_c_terminal(peptide, uniprot_data, uniprot_id, adjacent_lenght, amino_acids_possible_set):
    one_less, two_less, three_less = peptide[:-1], peptide[:-2], peptide[:-3]
    last_one, last_two, last_three = peptide[-1:], peptide[-2:], peptide[-3:]
    last_peptide_residue, large_peptide = None, None
    if one_less in uniprot_data[uniprot_id]:
        last_peptide_residue, large_peptide = get_neighbour_sequence(uniprot_data, uniprot_id, one_less, adjacent_lenght, amino_acids_possible_set, 1)
    elif two_less in uniprot_data[uniprot_id]:
        adjacent_sequences = uniprot_data[uniprot_id].split(two_less)
        if adjacent_sequences[1][1] == last_one:
             last_peptide_residue, large_peptide = get_neighbour_sequence(uniprot_data, uniprot_id, two_less, adjacent_lenght, amino_acids_possible_set, 2)
    elif three_less in uniprot_data[uniprot_id]:
        adjacent_sequences = uniprot_data[uniprot_id].split(three_less)
        if adjacent_sequences[1][1:3] == last_two:
            last_peptide_residue, large_peptide = get_neighbour_sequence(uniprot_data, uniprot_id, three_less, adjacent_lenght, amino_acids_possible_set, 3)
    else:
        last_peptide_residue, large_peptide = None, None
    return last_peptide_residue, large_peptide

def get_neighbour_sequence(uniprot_data, uniprot_id, peptide, adjacent_lenght, amino_acids_possible_set, penalty):
    adjacent_lenght += penalty
    adjacent_sequences = uniprot_data[uniprot_id].split(peptide)
    post_sequence = adjacent_sequences[1][:adjacent_lenght]
    if len(post_sequence) == adjacent_lenght:
        large_peptide = "".join(peptide + post_sequence)
        large_peptide_set = set(large_peptide)
        if len(large_peptide_set.difference(amino_acids_possible_set)) == 0:
            return peptide[-1], large_peptide
        else:
            return None, None
    else:
        return None, None

def set_creator_of_amino_acids(amino_acids_possible):
    amino_acids_possible_set = set()
    for amino_acid in amino_acids_possible:
        amino_acids_possible_set.add(amino_acid)
    return amino_acids_possible_set
