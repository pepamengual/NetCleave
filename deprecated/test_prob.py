import numpy as np

def prob(obs, tot, freq, n):
    fobs = obs / tot
    frandom = freq**n
    score = round(-np.log2(fobs/frandom), 3)
    print("{}: ({} / {}) - {}**{}".format(score, obs, tot, freq, n))

def main():
    obs = 1
    tot = 328000
    freq = 0.05
    n = 5
    prob(obs, tot, freq, n)
main()

