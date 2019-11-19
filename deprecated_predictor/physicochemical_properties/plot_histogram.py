def histogram_plotter(ms_score_list, random_score_list, name):
    import matplotlib.pyplot as plt
    import numpy as np

    print("{}: MS peptides {} / RANDOM peptides {}".format(name, len(ms_score_list), len(random_score_list)))
    
    TP = np.searchsorted(sorted(ms_score_list), 0, side="right") + 1
    FN = len(ms_score_list) - TP
    FP = np.searchsorted(sorted(random_score_list), 0, side="right") + 1
    TN = len(ms_score_list) - FP
    
    denom = (np.sqrt((TP + FP))*np.sqrt((TP + FN))*np.sqrt((TN + FP))*np.sqrt((TN + FN)))
    MCC = round(((TP * TN) - (FP * FN)) / denom, 2)

    print("{} MCC is {}. TP: {}, FP: {}, FN: {}, TN: {}".format(name, MCC, TP, FP, FN, TN))

    plt.hist(ms_score_list, bins=100, label="MS", alpha=0.5)
    plt.hist(random_score_list, bins=100, label="RANDOM", alpha=0.5)

    plt.xlabel("PROcleavage score")
    plt.ylabel("Number of peptides")
    plt.legend()
    plt.title("{} prediction".format(name))
    plt.savefig("{}_cleavage_physicochemical.png".format(name))
    #plt.show()
    plt.cla()
    plt.clf()

