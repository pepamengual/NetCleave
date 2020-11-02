import cython
#cython: language_level=3

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_peptidome(dict sequence_data):
    cdef set peptidome = set()
    cdef str i, sequence, chunk
    cdef int n = 7
    cdef int j, l
    cdef list chunks
    for i, sequence in sequence_data.items():
        l = len(sequence)
        chunks = [sequence[j:j+n] for j in range(0, l, n)]
        for chunk in chunks[:-1]: #avoid having peptides with less than n residues
            if not "X" in chunk:
                peptidome.add(chunk)
    return list(peptidome)
