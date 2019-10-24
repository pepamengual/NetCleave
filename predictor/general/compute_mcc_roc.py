
def mcc_roc_computer(scored_dict, plot_name):
    import numpy as np
    print("Computing MCC...")
    mcc_list = []
    score_list = []
    cleavaged_list = sorted(scored_dict["MS"])
    random_list = sorted(scored_dict["RANDOM"])

    tpr_list = []
    fpr_list = []

    for i in np.arange(-15, 15, 0.1):
        TP = len([x for x in cleavaged_list if x <= i])
        FP = len([x for x in random_list if x < i])
        FN = len([x for x in cleavaged_list if x >= i])
        TN = len([x for x in random_list if x > i])
        if TP != 0:
            TPR = TP /(TP + FN)
        else:
            TPR = 0
        if FP != 0:
            FPR = FP / (FP + TN)
        else:
            FPR = 0
        tpr_list.append(TPR)
        fpr_list.append(FPR)
        try:
            MCC = round(((TP * TN) - (FP * FN)) / ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN))**0.5, 2)
        except:
            MCC = 0
        mcc_list.append(MCC)
        score_list.append(i)
    max_mcc = max(mcc_list)
    max_index = mcc_list.index(max_mcc)
    max_score = score_list[max_index]
    print("Highest MCC is {} at {} score".format(max_mcc, max_score))

    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    print("Plotting roc curve...")

    roc_auc = round(metrics.auc(fpr_list, tpr_list), 2)
    plt.plot(fpr_list, tpr_list, 'b', label = "AUC = {}".format(roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("{}_roc.png".format(plot_name))
