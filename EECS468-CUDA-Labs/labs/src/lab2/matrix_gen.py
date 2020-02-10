"""
Generate matrices for testing CUDA tiling program.
"""

import numpy as np



#
row_m = 10
col_m = 10

row_n = 10
col_n = 10

M = 2*np.ones((row_m, col_m))
N = 5*np.ones((row_m, col_m))

print('M:', M, '\n\n')
#print('N:', N)

P = np.dot(M, N)

#print('P:', P)


# write M and N to files
f = open('testMatrix1.txt', 'a')

# write M
for row in M:
    for e in row:
        f.write(str(e) + ' ')
f.close()

# write N
f = open('testMatrix2.txt', 'a')
for row in N:
    for e in row:
        f.write(str(e) + ' ')

print('M * N = P ----- P: \n', P)
