def statistics(scored_dict, plot_name):
    """ Computing MCC for each threshold, then get the highest
    """
    mcc_dictionary, tpr_list, fpr_list = mcc_computer(scored_dict)
    max_mcc = max(mcc_dictionary.values())
    index = list(mcc_dictionary.keys())[list(mcc_dictionary.values()).index(max_mcc)]
    print("Highest MCC is {} at {} score".format(max_mcc, index))

    """ Computing ROC curve
    """
    roc_computer(tpr_list, fpr_list, plot_name)

def mcc_computer(scored_dict):
    import numpy as np
    print("Computing MCC...")
    mcc_dictionary = {}
    tpr_list = []
    fpr_list = []

    cleavaged_list = sorted(scored_dict["MS"])
    random_list = sorted(scored_dict["RANDOM"])
    all_list = cleavaged_list + random_list

    for i in np.arange(min(all_list) -0.1, max(all_list) + 0.1, 0.1):
        TP = np.searchsorted(cleavaged_list, i, side="right") + 1
        FN = len(cleavaged_list) - TP
        FP = np.searchsorted(random_list, i, side="right") + 1
        TN = len(random_list) - FP
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
        denom = (np.sqrt((TP + FP))*np.sqrt((TP + FN))*np.sqrt((TN + FP))*np.sqrt((TN + FN)))
        MCC = round(((TP * TN) - (FP * FN)) / denom, 2)
        if np.isinf(MCC):
            MCC = 0
        mcc_dictionary.setdefault(i, MCC)
    return mcc_dictionary, tpr_list, fpr_list

def roc_computer(tpr_list, fpr_list, plot_name):
    import sklearn.metrics as metrics
    import matplotlib.pyplot as plt
    print("Plotting roc curve...")
    roc_auc = round(metrics.auc(fpr_list, tpr_list), 2)
    print("AUC value is {}".format(roc_auc))
    plt.plot(fpr_list, tpr_list, 'b', label = "AUC = {}".format(roc_auc))
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()
    plt.savefig("{}_roc.png".format(plot_name))
    plt.clf()
    plt.cla()
