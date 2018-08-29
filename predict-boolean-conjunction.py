# This code learns the boolean formula from examples.
# Implemented by the Consistency Algorithm.

import numpy as np
import sys

training_examples = np.loadtxt(sys.argv[1], int)

rows = training_examples.shape[0]
cols = training_examples.shape[1]
d = cols - 1

x = np.delete(training_examples, -1, axis=1)  # the matrix is without last col
y = training_examples[:, d]  # y tags

# The Consistency Algorithm
h_pos = np.ones(d, int)
h_neg = np.ones(d, int)

for i in range(0, rows):  # for instance t in examples do
    if y[i] == 1:
        # calculating y_head
        y_head = 1
        for j in range(0, d):
            if x[i][j] == 1:  # if 1 we remove not(xi)
                if h_neg[j] == 1:
                    y_head = 0
                    break
            elif x[i][j] == 0:  # if 0 we remove (xi)
                if h_pos[j] == 1:
                    y_head = 0
                    break
        if y_head == 0:  # our hypothesis isn't good anymore.
            for j in range(0, d):
                if x[i][j] == 1:
                    h_neg[j] = 0
                elif x[i][j] == 0:
                    h_pos[j] = 0
formula = ""  # output format
for i in range(0, d):
    if h_pos[i] == 1:
        formula = formula + "x" + str(i + 1) + ","
    if h_neg[i] == 1:
        formula = formula + "not(x" + str(i + 1) + "),"
if len(formula) > 0:
    formula = formula[: -1]

my_file = "output.txt"
o_file = open(my_file, 'w')
o_file.write(formula)  # output
o_file.close()
