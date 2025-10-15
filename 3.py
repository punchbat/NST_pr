import numpy as np

def vector_to_binary_matrix(vec: np.ndarray, width: int = None) -> np.ndarray:
    vec = vec.astype(int)
    max_val = np.max(vec)
    if width is None:
        width = len(bin(max_val)) - 2 
    
    bin_matrix = np.array([
        [int(bit) for bit in format(num, f'0{width}b')]
        for num in vec
    ])
    return bin_matrix

import numpy as np

def unique_rows(matrix: np.ndarray) -> np.ndarray:
    return np.unique(matrix, axis=0)


m = np.array([
    [1, 0, 1],
    [1, 1, 1],
    [1, 0, 1],
    [0, 0, 0]
])
print(unique_rows(m))

v = np.array([3, 5, 7])
print(vector_to_binary_matrix(v))