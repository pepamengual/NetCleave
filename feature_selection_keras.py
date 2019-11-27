# Feature Extraction with RFE
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


def read_data(data_file):
    training_table = pd.read_csv(data_file, index_col=0)
    return training_table

def feature_extraction(training_table, first, second):
    Y = training_table["class"]
    l = ["P{}".format(i) for i in range(first, second)]
    X = training_table[l]
    
    print("Initializing model")
    model = LogisticRegression(solver='lbfgs', max_iter=4000)
    print("Model initialized")
    rfe = RFE(model, 20)
    print("RFE, fitting")
    fit = rfe.fit(X, Y)
    
    #print("Num Features: %d" % fit.n_features_)
    #print("Selected Features: %s" % fit.support_)
    #print("Feature Ranking: %s" % fit.ranking_)
    
    ranking = fit.ranking_
    
    ranking = list(ranking)
    selected_features = []
    for i, rank in enumerate(ranking):
        if rank == 1:
            selected_features.append(l[i])
    return selected_features

def main():
    data_file = "20k_rows_ml_aaindex.csv"
    training_table = read_data(data_file)

    f0 = [(0, 553), (553, 1106), (1106, 1659), (1659, 2212), (2212, 2765), (2765, 3318)]

    all_selected_features = []
    for tupple in f0:
        first = tupple[0]
        second = tupple[1]
        selected_features = feature_extraction(training_table, first, second)
        all_selected_features.extend(selected_features)

    print(all_selected_features)
main()
