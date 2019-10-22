
def mcc_computer(scored_dict):
    import numpy as np
    print("Computing MCC...")
    mcc_list = []
    score_list = []
    cleavaged_list = sorted(scored_dict["Cleavage sites"])
    random_list = sorted(scored_dict["Random sites"])

    for i in np.arange(-5, 0, 0.1):
        TP = len([x for x in cleavaged_list if x <= i])
        FP = len([x for x in random_list if x < i])
        FN = len([x for x in cleavaged_list if x >= i])
        TN = len([x for x in random_list if x > i])
        MCC = round(((TP * TN) - (FP * FN)) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))**0.5, 2)
        mcc_list.append(MCC)
        score_list.append(i)
    max_mcc = max(mcc_list)
    max_index = mcc_list.index(max_mcc)
    max_score = score_list[max_index]
    print("Highest MCC is {} at {} score".format(max_mcc, max_score))
