import pandas as pd
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

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

def normalize_descriptors_table(df):
    print("---> Normalizing descriptors...")
    df_normalized=(df-df.min())/(df.max()-df.min())
    return df_normalized

def encode_sequence_data(sequence_table, df_normalized):
    print("---> Encoding data using the descriptors...")
    encode_map, encode_data = {}, []
    for r in list("ACDEFGHIKLMNPQRSTVWY"):
        encode_map.setdefault(r, df_normalized.loc[r].tolist())

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
    encoded_labeled_df = pd.concat([encoded_df, class_table], axis=1)
    encoded_labeled_df = encoded_labeled_df.sample(frac=1).reset_index(drop=True) #shuffles data
    return encoded_labeled_df

def run_feature_importance(encoded_labeled_df):
    print("---> Running feature importance analysis...")
    features = encoded_labeled_df.drop(["class"], axis=1)
    classes = encoded_labeled_df["class"]

    model = XGBClassifier()
    model.fit(features, classes)

    feature_importance = model.feature_importances_
    feature_names = features.columns.tolist()
    for name, score in zip(feature_names, feature_importance):
        print("Feature: {0}, Score: {1}".format(name, score))
    
    plt.bar(feature_names, feature_importance)
    plt.tight_layout()
    plt.savefig("feature_selection_QSAR_xdgboost.png", dpi=300)   

#def feature_selection(training_data_path, model_path):
def feature_selection():
    #training_data_path = "../../data/training_data/I_mass-spectrometry/I_mass-spectrometry_sequence_class.txt"
    training_data_path = "sample_data.txt"
    peptide_lenght = 7
    sequence_table, class_table = read_data_table(training_data_path)

    #descriptors_path = "predictor/ml_main/QSAR_table.csv"
    descriptors_path = "VHSE_table.csv"
    df_descriptors = read_descriptors_table(descriptors_path)
    df_normalized = normalize_descriptors_table(df_descriptors)

    encode_data = encode_sequence_data(sequence_table, df_normalized)
    encoded_df = generate_encoded_df(encode_data, peptide_lenght, df_descriptors)
    
    encoded_labeled_df = generate_encoded_labeled_df(encoded_df, class_table)
    run_feature_importance(encoded_labeled_df)

feature_selection()
