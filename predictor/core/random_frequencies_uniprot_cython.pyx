import cython
#cython: language_level=3

def generate_empty_dictionary(residues):
    counts = {}
    for residue in residues:
        counts.setdefault(residue, 0)
    return counts

def generate_frequencies(counts):
    frequencies = {}
    total_counts = sum(list(counts.values()))
    for residue, counts_residue in counts.items():
        frequency = counts_residue/total_counts
        frequencies.setdefault(residue, frequency)
    return frequencies

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_random_frequencies(dict uniprot_data):
    print("Computing random frequencies from Uniprot and Uniparc")
    cdef str residues = "ACDEFGHIKLMNPQRSTVWY"
    cdef dict counts, frequencies
    cdef str uniprot_id, sequence, amino_acid
    counts = generate_empty_dictionary(residues)
    for uniprot_id, sequence in uniprot_data.items():
        for amino_acid in sequence:
            if amino_acid in counts:
                counts[amino_acid] += 1

    frequencies = generate_frequencies(counts)
    return frequencies
