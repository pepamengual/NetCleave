import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr
from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from predictor.ml_main.ml_utilities import integer_encoding
from keras.backend.tensorflow_backend import tf
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
graph = tf.get_default_graph()
import pandas as pd
import numpy as np
import pickle

def read_uniprot_pickle(uniprot_pickle_path):
    with open(uniprot_pickle_path, "rb") as f:
        uniprot_pickle = pickle.load(f)
    return uniprot_pickle

def read_input_file(input_file):
    df = pd.read_csv(input_file, skiprows=[0])
    return df

def load_model(model_file):
    neurons = 200
    model = Sequential()
    model.add(Dense(int(neurons*2), input_dim=neurons, activation="sigmoid")) # Hidden Layer 1 that receives the Input from the Input Layer
    model.add(Dense(int(neurons), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(int(neurons/2), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(int(neurons/4), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(1, activation='sigmoid')) #Output layer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(model_file)
    return model

def encode_candidates(df):
    encode_df = df
    encoding_table = integer_encoding.integer_encoding(encode_df)
    max_length = 10
    padding_table = pad_sequences(encoding_table, maxlen=max_length, padding='post', truncating='post')
    one_hot_table = to_categorical(padding_table, num_classes=20)
    train_ohe = one_hot_table.reshape(encode_df.shape[0], 1, max_length*20)
    train_ohe = train_ohe.astype(int)
    train_ohe = train_ohe.tolist()
    train_ohe_list = []
    for i in train_ohe:
        for j in i:
            train_ohe_list.append(j)
    one_hot_df = pd.DataFrame(train_ohe_list)
    return one_hot_df

def parse_df(df):
    mel8_unmodified = df.loc[(df["Intensity Mel-8_HLA-I (arbitrary units)"] >= 0) & (df["Modifications"] == "Unmodified")]
    mel15_unmodified = df.loc[(df["Intensity Mel-15_HLA-I (arbitrary units)"] >= 0) & (df["Modifications"] == "Unmodified")]
    mel16_unmodified = df.loc[(df["Intensity Mel-16_HLA-I (arbitrary units)"] >= 0) & (df["Modifications"] == "Unmodified")]
    return mel8_unmodified, mel15_unmodified, mel16_unmodified

def remove_redundancy(df, uniprot_pickle):
    peptide_selection = []
    uniprot_selection = []
    fasta_selection = []
    long_peptides_selection = []
    proteasome_selection = []

    redundant_uniprot = df["Proteins"].tolist()
    reduntant_peptide = df["Sequence"].tolist()

    for uniprot_entry, peptide_entry in zip(redundant_uniprot, reduntant_peptide):
        if type(uniprot_entry) != float:
            list_of_uniprot_ids = uniprot_entry.split(";")
            set_of_uniprot_ids = set()
            for uniprot_id_full in list_of_uniprot_ids:
                uniprot_id = uniprot_id_full.split("-")[0]
                set_of_uniprot_ids.add(uniprot_id)
            
            list_long_peptides = []
            list_proteasome_region = []
            new_list_uniprot_ids = []
            list_of_sequences = []
            for uniprot_id in set_of_uniprot_ids:
                if uniprot_id in uniprot_pickle:
                    sequence = uniprot_pickle[uniprot_id]
                    if peptide_entry in sequence: # finding for non-mutated peptides
                        index_begining = sequence.index(peptide_entry)
                        index_end = index_begining + len(peptide_entry)
                        peptide_check = sequence[index_begining:index_end]
                        if peptide_entry == peptide_check:
                            after_cleavage_proteasome = sequence[index_end:index_end+5]
                            if len(after_cleavage_proteasome) == 5 and len(peptide_entry) == 9: #here
                                long_peptide = "{}{}".format(peptide_entry, after_cleavage_proteasome)
                                proteasome_region = "{}{}".format(peptide_entry[-5:], after_cleavage_proteasome)
                                list_long_peptides.append(long_peptide)
                                list_proteasome_region.append(proteasome_region)
                                new_list_uniprot_ids.append(uniprot_id)
                                list_of_sequences.append(sequence)
            
            if new_list_uniprot_ids and list_of_sequences and list_long_peptides and len(set(list_long_peptides)) == 1 and list_long_peptides:
                peptide_selection.append(peptide_entry)
                uniprot_selection.append(new_list_uniprot_ids)
                fasta_selection.append(list_of_sequences)
                long_peptides_selection.append(list(set(list_long_peptides))[0])
                proteasome_selection.append(list(set(list_proteasome_region))[0])
    return peptide_selection, uniprot_selection, fasta_selection, long_peptides_selection, proteasome_selection

def build_new_df(mel8_unmodified, mel15_unmodified, mel16_unmodified, uniprot_pickle):
    peptide_selection_mel8, uniprot_selection_mel8, fasta_selection_mel8, long_selection_mel8, proteasome_selection_mel8 = remove_redundancy(mel8_unmodified, uniprot_pickle)
    peptide_selection_mel15, uniprot_selection_mel15, fasta_selection_mel15, long_selection_mel15, proteasome_selection_mel15 = remove_redundancy(mel15_unmodified, uniprot_pickle)
    peptide_selection_mel16, uniprot_selection_mel16, fasta_selection_mel16, long_selection_mel16, proteasome_selection_mel16 = remove_redundancy(mel16_unmodified, uniprot_pickle)

    df8 = pd.DataFrame({"mel8_sequence": peptide_selection_mel8, "uniprot_list": uniprot_selection_mel8, "sequence": fasta_selection_mel8, "long_sequence": long_selection_mel8, "proteasome": proteasome_selection_mel8})
    df15 = pd.DataFrame({"mel15_sequence": peptide_selection_mel15, "uniprot_list": uniprot_selection_mel15, "sequence": fasta_selection_mel15, "long_sequence": long_selection_mel15, "proteasome": proteasome_selection_mel15})
    df16 = pd.DataFrame({"mel16_sequence": peptide_selection_mel16, "uniprot_list": uniprot_selection_mel16, "sequence": fasta_selection_mel16, "long_sequence": long_selection_mel16, "proteasome": proteasome_selection_mel16})
    return df8, df15, df16

def printer_classes(df, string):
    cleavage = df.loc[df["prediction"] == 1]
    non_cleavage = df.loc[df["prediction"] == 0]
    list_cleavage = len(cleavage["prediction"].tolist())
    list_non_cleavage = len(non_cleavage["prediction"].tolist())
    print("{}: {} of {}; {}".format(string, list_cleavage, list_cleavage + list_non_cleavage, round(list_cleavage/(list_cleavage + list_non_cleavage), 2)))

def printer_predictions(df, string):
    cleavage = df.loc[df["prediction"] >= 0.5]
    non_cleavage = df.loc[df["prediction"] < 0.5]
    list_cleavage = len(cleavage["prediction"].tolist())
    list_non_cleavage = len(non_cleavage["prediction"].tolist())
    print("{}: {} of {}; {}".format(string, list_cleavage, list_cleavage + list_non_cleavage, round(list_cleavage/(list_cleavage + list_non_cleavage), 2)))


def predictor(df8, df15, df16, proteasome_model):
    one_hot_df_df8 = encode_candidates(df8)
    one_hot_df_df15 = encode_candidates(df15)
    one_hot_df_df16 = encode_candidates(df16)
    
    prediction_df8 = proteasome_model.predict(one_hot_df_df8)
    prediction_df15 = proteasome_model.predict(one_hot_df_df15)
    prediction_df16 = proteasome_model.predict(one_hot_df_df16)


   #prediction_df8 = proteasome_model.predict_classes(one_hot_df_df8)
   #prediction_df15 = proteasome_model.predict_classes(one_hot_df_df15)
   #prediction_df16 = proteasome_model.predict_classes(one_hot_df_df16)
    
    prediction_df_df8 = pd.DataFrame(prediction_df8, columns=["prediction"])
    prediction_df_df15 = pd.DataFrame(prediction_df15, columns=["prediction"])
    prediction_df_df16 = pd.DataFrame(prediction_df16, columns=["prediction"])

    proteasome_df_df8 = pd.concat([df8, prediction_df_df8], axis=1)
    proteasome_df_df15 = pd.concat([df15, prediction_df_df15], axis=1)
    proteasome_df_df16 = pd.concat([df16, prediction_df_df16], axis=1)
    
   #printer_classes(proteasome_df_df8, "mel8")
   #printer_classes(proteasome_df_df15, "mel15")
   #printer_classes(proteasome_df_df16, "mel16")
    print(proteasome_df_df15)
    import matplotlib.pyplot as plt
    plt.hist(proteasome_df_df8["prediction"], bins=50, label="mel8", alpha=0.2)
    plt.hist(proteasome_df_df15["prediction"], bins=50, label="mel15", alpha=0.2)
    plt.hist(proteasome_df_df16["prediction"], bins=50, label="mel16", alpha=0.2)
    plt.legend()
    plt.show()

    printer_predictions(proteasome_df_df8, "mel8")
    printer_predictions(proteasome_df_df15, "mel15")
    printer_predictions(proteasome_df_df16, "mel16")

   #save_list = []
    mel8_list = proteasome_df_df8["mel8_sequence"].tolist()
    mel15_list = proteasome_df_df15["mel15_sequence"].tolist()
    mel16_list = proteasome_df_df16["mel16_sequence"].tolist()
   #save_list.extend(mel8_list)
   #save_list.extend(mel15_list)
   #save_list.extend(mel16_list)

    l = [mel8_list, mel15_list, mel16_list]
    l_name = ["mel8_list", "mel15_list", "mel16_list"]

    for l_list, name in zip(l, l_name):
        with open("{}.txt".format(name), "w") as f:
            for peptide in l_list:
                f.write(peptide[4] + "\n")


   #save_list = set(save_list)
   #with open("testing_peptides.txt", "w") as f:
   #    for peptide in save_list:
   #        f.write(peptide + "\n")


def main():
    uniprot_pickle_path = "../../../data/parsed/uniprot/uniprot_sequences.pickle"
    uniprot_pickle = read_uniprot_pickle(uniprot_pickle_path)
    
    input_file = "41467_2016_BFncomms13404_MOESM1318_ESM.csv"
    df = read_input_file(input_file)
    
    #proteasome_file = "../../model_proteasome.h5"
    proteasome_file = "model_proteasome.h5"
    proteasome_model = load_model(proteasome_file)

    mel8_unmodified, mel15_unmodified, mel16_unmodified = parse_df(df)
    df8, df15, df16 = build_new_df(mel8_unmodified, mel15_unmodified, mel16_unmodified, uniprot_pickle)

    predictor(df8, df15, df16, proteasome_model)

main()
