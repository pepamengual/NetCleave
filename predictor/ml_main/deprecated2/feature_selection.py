import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

def read_data_table(path):
    print("---> Reading training data...")
    training_data_file = path #delete for final version
    #training_data_file = "{}/{}_sequence_class.txt".format(path, path.split("/")[-1])
    df = pd.read_csv(training_data_file, sep="\t", index_col=None, header=0)
    sequence_table = df.drop(['class'], axis=1)
    class_table = df['class']
    return sequence_table, class_table

def read_descriptors_table(path):
    print("---> Reading descriptors...")
    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    return df

def encode_sequence_data(sequence_table, df):
    print("---> Encoding data using the descriptors...")
    encode_map, encode_data = {}, []
    for r in list("ACDEFGHIKLMNPQRSTVWY"):
        encode_map.setdefault(r, df.loc[r].tolist())

    for sequence in sequence_table['sequence'].values:
        sequence_encode = []
        for r in sequence:
            sequence_encode.extend(encode_map[r])
        encode_data.append(sequence_encode)
    return encode_data

def generate_encoded_df(encode_data, peptide_lenght, df):
    print("---> Generating a descriptor dataframe...")
    descriptor_header = df.columns.tolist()
    encoded_df = pd.DataFrame(encode_data, columns=["{}_{}".format(i, j) for i in range(peptide_lenght) for j in descriptor_header])
    return encoded_df

def generate_encoded_labeled_df(encoded_df, class_table):
    print("---> Labeling the descriptor dataframe...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(encoded_df)
    scaled_df = pd.DataFrame(scaled_data, columns=encoded_df.columns)

    encoded_labeled_df = pd.concat([scaled_df, class_table], axis=1)
    encoded_labeled_df = encoded_labeled_df.sample(frac=1).reset_index(drop=True) #shuffles data
    return encoded_labeled_df

def run_feature_importance(encoded_labeled_df):
    print("---> Running feature importance analysis...")
    features = encoded_labeled_df.drop(["class"], axis=1)
    classes = encoded_labeled_df["class"]
    
    X_train, X_test, y_train, y_test = train_test_split(features, classes, random_state=42, test_size=.33)
    
    model = XGBClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
       
    data_importance = {}
    for feature_name, feature_value in zip(X_train.columns, model.feature_importances_):
        data_importance.setdefault(feature_name, feature_value)    
    thresholds_sorted = sorted(data_importance.values())
    features_sorted = [k for k, v in sorted(data_importance.items(), key=lambda item: item[1])]

    fig, ax = plt.subplots(figsize=(10,10))
    ylocs, values = features_sorted, thresholds_sorted
    ax.barh(ylocs, values, align="center")
    for x, y in zip(values, ylocs):
        ax.text(x, y, round(x, 3), va='center')
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("Features")
    fig.tight_layout()
    fig.savefig("feature_importance_test_VHSE.png", bbox_inches="tight", dpi=300)
    
    higher_accuracy, selected_features = 0, []
    for i, value in enumerate(thresholds_sorted):
        if value >= 0.005:
            feature_names = features_sorted[i:]
            # select features using threshold
            selection = SelectFromModel(model, threshold=value, prefit=True)
            select_X_train = selection.transform(X_train)
            # train model
            selection_model = XGBClassifier()
            selection_model.fit(select_X_train, y_train)
            # eval model
            select_X_test = selection.transform(X_test)
            y_pred = selection_model.predict(select_X_test)
            predictions = [round(value) for value in y_pred]
            accuracy = accuracy_score(y_test, predictions)
            accuracy = round(accuracy, 4)
            print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (value, select_X_train.shape[1], accuracy*100.0))
            print(feature_names)
            if accuracy > higher_accuracy:
                higher_accuracy = accuracy
                selected_features = feature_names

    print("The best features with an accuracy of {} and n={} are...".format(higher_accuracy, len(selected_features)))
    print(selected_features)
    

#def feature_selection(training_data_path, model_path):
def feature_selection():
    training_data_path = "../../data/training_data/I_mass-spectrometry/I_mass-spectrometry_sequence_class.txt"
    #training_data_path = "sample_data.txt"
    peptide_lenght = 7
    sequence_table, class_table = read_data_table(training_data_path)

    #descriptors_path = "predictor/ml_main/QSAR_table.csv"
    descriptors_path = "VHSE_table.csv"
    df_descriptors = read_descriptors_table(descriptors_path)

    encode_data = encode_sequence_data(sequence_table, df_descriptors)
    encoded_df = generate_encoded_df(encode_data, peptide_lenght, df_descriptors)
    
    encoded_labeled_df = generate_encoded_labeled_df(encoded_df, class_table)
    run_feature_importance(encoded_labeled_df)

feature_selection()
