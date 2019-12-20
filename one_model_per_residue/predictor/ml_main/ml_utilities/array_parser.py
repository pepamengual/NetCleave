import numpy as np
import itertools

def array_parser(numbers):
    result = []
    for peptido in numbers:
        result1 = list(np.array(peptido).flat)
        #result1 = list(itertools.chain.from_iterable(peptido))
        #result1 = [x for b in peptido for x in b]
        result.append(result1)
    result_np = np.array(result)
    return result_np
