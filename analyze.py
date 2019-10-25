import numpy as np
import matplotlib.pyplot as plt
d = {}

with open("proteasome_scores.txt", "r") as f:
    for line in f:
        line = line.rstrip().split()
        score = float(line[0])
        sequence = line[1]
        if len(sequence) == 5:
            c_term_proteasome = sequence[2:5]
            d.setdefault(c_term_proteasome, []).append(score)
d2 = {}

for c_term_proteasome, score_list in d.items():
    mean = round(np.mean(score_list), 3)
    d2.setdefault(c_term_proteasome, (len(score_list), mean))

with open("last_three_proteasome.txt", "w") as f:
    for c_term_proteasome, tup in sorted(d2.items(), key=lambda kv: kv[1]):
        mean = tup[0]
        n = tup[1]
        line = "{} {} {}".format(c_term_proteasome, mean, n)
        f.write(line + "\n")

print("Plotting now")
for c_term_proteasome, tup in sorted(d2.items(), key=lambda kv: kv[1]):
    mean = tup[0]
    n = tup[1]
    plt.scatter(n, mean)
plt.show()
