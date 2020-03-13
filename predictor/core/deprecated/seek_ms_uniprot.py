def seeking_ms(ms_data, uniprot_data, adjacent_lenght):
    total_peptides = 0
    found_in_uniprot = 0
    not_found_in_uniprot = 0
    found_in_uniprot_and_sequence = 0
    found_in_uniprot_not_sequence = 0
    having_adjacent = 0
    
    amino_acids_possible = "ACDEFGHIKLMNPQRSTVWY"
    amino_acids_possible_set = set()
    for amino_acid in amino_acids_possible:
        amino_acids_possible_set.add(amino_acid)

    data = []
    list_of_used_peptides = []
    for uniprot_id, peptide_set in ms_data.items():
        for peptide in peptide_set:
            total_peptides += 1
            if uniprot_id in uniprot_data:
                found_in_uniprot += 1
                if peptide in uniprot_data[uniprot_id]:
                    found_in_uniprot_and_sequence += 1
                    adjacent_sequences = uniprot_data[uniprot_id].split(peptide)
                    pre_sequence = adjacent_sequences[0][-adjacent_lenght:]
                    post_sequence = adjacent_sequences[1][:adjacent_lenght]
                    if len(pre_sequence) == adjacent_lenght and len(post_sequence) == adjacent_lenght:
                        large_peptide = "".join(pre_sequence + peptide + post_sequence)
                        large_peptide_set = set(large_peptide)
                        if len(large_peptide_set.difference(amino_acids_possible_set)) == 0:
                            data.append(large_peptide)
                            list_of_used_peptides.append(peptide)
                            having_adjacent += 1
                else:
                    found_in_uniprot_not_sequence += 1
            else:
                not_found_in_uniprot += 1
    print("\n")
    print("Results summary...")
    print("{} / {} peptides have been found in UniProt.".format(found_in_uniprot_and_sequence, total_peptides))
    print("{} peptides have deprecated UniProt ID.".format(not_found_in_uniprot))
    print("{} peptides could not be located in its UniProt sequence.".format(found_in_uniprot_not_sequence))
    print("{} / {} peptides have both adjacent sequences of lenght {}.".format(having_adjacent, found_in_uniprot_and_sequence, adjacent_lenght))

    return data, list_of_used_peptides
