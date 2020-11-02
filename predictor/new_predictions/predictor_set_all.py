import tensorflow as tf
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
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False
logger = tf.get_logger()
logger.disabled = True
logger.setLevel(logging.FATAL)
#graph = tf.get_default_graph()
import random
import pandas as pd
import numpy as np
from keras import regularizers
from keras import backend as K
import matplotlib.pyplot as plt

def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
    """
    char_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
                 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
                 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    encode_list = []
    for row in data['sequence'].values:
        row_encode = []
        for i, code in enumerate(row):
            row_encode.append(char_dict.get(code))
        encode_list.append(np.array(row_encode))
    return encode_list

def load_model(model_file, max_lenght):
    neurons = max_lenght*20 #changed to remove middle
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(int(neurons/3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.load_weights(model_file)
    return model

def encode_candidates(list_of_candidates, max_lenght):
    encode_df = pd.DataFrame({"sequence": list_of_candidates})
    encoding_table = integer_encoding(encode_df)
    padding_table = pad_sequences(encoding_table, maxlen=max_lenght, padding='post', truncating='post')
    one_hot_table = to_categorical(padding_table, num_classes=20)
    train_ohe = one_hot_table.reshape(encode_df.shape[0], 1, max_lenght*20)
    train_ohe = train_ohe.astype(int)
    train_ohe = train_ohe.tolist()
    train_ohe_list = []
    for i in train_ohe:
        for j in i:
            train_ohe_list.append(j)
    one_hot_df = pd.DataFrame(train_ohe_list)
    return one_hot_df

def proteasome_prediction(score_set_path, models_path, cell_type, name):
    max_lenght = 7
    score_set_df = pd.read_csv(score_set_path, sep="\t")
    score_list = []
    score_dict = {}
    proteasome_file = "{}/proteasome_all_models.h5".format(models_path)
    proteasome_model = load_model(proteasome_file, max_lenght)
    for amino_acid in sorted(set(score_set_df["residue"])):
        score_set_df_amino_acid = score_set_df.loc[score_set_df["residue"] == amino_acid]
        list_of_candidates = list(score_set_df_amino_acid["prediction_region"])
        list_of_peptides = list(score_set_df_amino_acid["peptide"])
        
        one_hot_df = encode_candidates(list_of_candidates, max_lenght) #one hot encode candidates
        #prediction = proteasome_model.predict_classes(one_hot_df) #make prediction of candidates (0 no cleavage, 1 cleavage)
        prediction = proteasome_model.predict(one_hot_df)
        candidate_df = pd.DataFrame({"sequence": list_of_candidates}) #a dataframe of the predicted sequences
        peptide_df = pd.DataFrame({"peptide": list_of_peptides}) #a dataframe of the cleavage position (+1) of the predicted sequences
        prediction_df = pd.DataFrame(prediction, columns=["prediction"]) #a dataframe of the predicted cleavages (0 no cleavage, 1 cleavage)
        proteasome_df = pd.concat([candidate_df, peptide_df, prediction_df], axis=1) #concatenate the above dataframes into a single one
        
        #partial_proteasome_df = pd.concat([candidate_df, prediction_df], axis=1)
        #partial_proteasome_df.drop_duplicates(keep="first", inplace=True)
        
        score_list.extend(list(proteasome_df["prediction"]))
        score_dict.setdefault(amino_acid, list(proteasome_df["prediction"]))
        #proteasome_cleavage_regions = proteasome_df.loc[proteasome_df["prediction"] >= 0.5] #select rows with positive proteasome predicted cleavage
    
    plot_boxplots(score_dict, cell_type, name)
    plot_prediction_per_residue_type(score_dict, cell_type, name)
    plot_histogram(score_list, name)


def plot_boxplots(score_dict, cell_type, name):
    labels, data = [*zip(*score_dict.items())]  # 'transpose' items to parallel key, value lists
    labels_lenght = []
    for label in labels:
        new_label = "{}_{}".format(label, len(score_dict[label]))
        labels_lenght.append(new_label)

    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels_lenght)
    plt.title(cell_type)
    plt.tight_layout()
    plt.savefig("{}_boxplots.png".format(name))
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

def plot_prediction_per_residue_type(score_dict, cell_type, name):
    for aa, values in sorted(score_dict.items()):
        total = len(values)
        counts = []
        for i in np.arange(0, 1.05, 0.05):
            i = round(i, 2)
            less_than_i = sum(j <= i for j in values)
            counts.append(less_than_i)
        score_numbers = [round(count/total, 2) if count > 0 else 0 for count in counts]
        score_thresholds = [round(i, 2) for i in np.arange(0, 1.05, 0.05)]
        #print(aa)
        #print(score_numbers)
        #print(score_thresholds)
        plt.plot(score_thresholds, score_numbers, label="{}_{}".format(aa, total))
    plt.title(cell_type)
    plt.legend()
    plt.tight_layout()
    plt.savefig("{}_lines.png".format(name))
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

def plot_histogram(score_list, name):
    greater_than_half = sum(i >= 0.5 for i in score_list)
    print("{}/{} peptides had a cleavage score >= 0.5".format(greater_than_half, len(score_list)))
    plt.hist(score_list, bins=20)
    plt.ylabel("Density")
    plt.xlabel("Probability")
    plt.savefig("{}_histogram.png".format(name))
    plt.show()
    plt.close()
    plt.cla()
    plt.clf()

