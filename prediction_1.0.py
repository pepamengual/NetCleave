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
import random
import pandas as pd
import numpy as np


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

def encode_candidates(list_of_candidates):
    encode_df = pd.DataFrame({"sequence": list_of_candidates})
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

def slicing_through_sequence(sequence):
    list_of_candidates = []
    list_of_positions = []
    for position, amino_acid in enumerate(sequence):
        candidate = sequence[position - 4: position + 6]
        if len(candidate) == 10:
            list_of_candidates.append(candidate)
            list_of_positions.append(position)
    return list_of_candidates, list_of_positions

def proteasome_prediction(sequence, proteasome_model):
    list_of_candidates, list_of_positions = slicing_through_sequence(sequence) #slice
    one_hot_df = encode_candidates(list_of_candidates) #one hot encode candidates
    prediction = proteasome_model.predict_classes(one_hot_df) #make prediction of candidates (0 no cleavage, 1 cleavage)

    candidate_df = pd.DataFrame({"sequence": list_of_candidates}) #a dataframe of the predicted sequences
    position_df = pd.DataFrame({"position": list_of_positions}) #a dataframe of the cleavage position (+1) of the predicted sequences
    prediction_df = pd.DataFrame(prediction, columns=["prediction"]) #a dataframe of the predicted cleavages (0 no cleavage, 1 cleavage)
    proteasome_df = pd.concat([candidate_df, position_df, prediction_df], axis=1) #concatenate the above dataframes into a single one
    print(proteasome_df)   
    proteasome_cleavage_regions = proteasome_df.loc[proteasome_df["prediction"] == 1] #select rows with positive proteasome predicted cleavage
    list_of_proteasome_cleavage = proteasome_cleavage_regions["position"].tolist() #a list of the positions of the sequence that are predicted to be cleaved by the proteosome
    list_of_proteasome_cleavage.extend([0, len(sequence) - 1]) #add initial position and last position of the sequence
    list_of_proteasome_cleavage.sort()
    return list_of_proteasome_cleavage

def peptides_cleaved_by_proteasome(sequence,list_of_proteasome_cleavage):
    list_of_peptides_cleaved_by_proteasome = []
    for position in list_of_proteasome_cleavage:
        list_of_proteasome_chunks = [i for i in list_of_proteasome_cleavage if i < position]
        for cleavage_possibility in list_of_proteasome_chunks:
            proteasome_chunk = sequence[cleavage_possibility : position + 1]
            #if len(proteasome_chunk) >= 9 and len(proteasome_chunk) <= 16:
            if len(proteasome_chunk) >= 8:
                list_of_peptides_cleaved_by_proteasome.append(proteasome_chunk)
                #print(cleavage_possibility, position, sequence[position], proteasome_chunk)
    return list_of_peptides_cleaved_by_proteasome

def hypothetical_peptides(list_of_peptides_cleaved_by_proteasome, sequence):
    print("Summary report\n")
    print("9mer possibilities from your sequence: {}".format(len(sequence)))
    peptide_set = set()
    for peptide in list_of_peptides_cleaved_by_proteasome:
        peptide_chosen = peptide[-9:]
        peptide_set.add(peptide_chosen)
    peptide_list = sorted(list(peptide_set))
    print("9mer predicted by proteasome cleavage: {}\n".format(len(peptide_list)))
    print(peptide_list)   

def main():
    proteasome_file = "model_proteasome.h5"
    proteasome_model = load_model(proteasome_file)
    
    sequence = "LVVSFVVGGLAVILPPLSPYFKYSVMINKATP"

    list_of_proteasome_cleavage = proteasome_prediction(sequence, proteasome_model)
    list_of_peptides_cleaved_by_proteasome = peptides_cleaved_by_proteasome(sequence, list_of_proteasome_cleavage)

    hypothetical_peptides(list_of_peptides_cleaved_by_proteasome, sequence)
main()
