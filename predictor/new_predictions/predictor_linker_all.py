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
    neurons = max_lenght*20
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

def slicing_through_sequence(sequence):
    candidate_dictionary = {}
    for position, amino_acid in enumerate(sequence):
        candidate = sequence[position - 3: position + 4]
        if len(candidate) == 7:
            candidate_dictionary.setdefault(amino_acid, {}).setdefault("candidates", []).append(candidate)
            candidate_dictionary.setdefault(amino_acid, {}).setdefault("positions", []).append(position)
    return candidate_dictionary

def peptides_cleaved_by_proteasome(sequence, all_proteasome_predictions, min_peptide_lenght_list, data_probabilities, peptide, linker):
    list_of_peptides_cleaved_by_proteasome = []
    possible_cleavages = [(i,round(j,4)) for i, j in sorted(data_probabilities.items()) if i <= len(peptide)+len(linker)+2]
    for amino_acid, list_of_proteasome_cleavage in sorted(all_proteasome_predictions.items()):
        for position in list_of_proteasome_cleavage:
            list_of_proteasome_chunks = [i for i in list_of_proteasome_cleavage if i < position]
            for cleavage_possibility in list_of_proteasome_chunks:
                proteasome_chunk = sequence[cleavage_possibility : position + 1]
                if len(proteasome_chunk) >= min_peptide_lenght_list: #edit here if needed
                    probability = round(data_probabilities[position], 4)
                    list_of_peptides_cleaved_by_proteasome.append((proteasome_chunk, probability))
    return list_of_peptides_cleaved_by_proteasome, possible_cleavages

def hypothetical_peptides(sequence, list_of_peptides_cleaved_by_proteasome, peptide_lenght_list):
    print("\nSummary report\n")
    possible_peptides = 0
    for lenght in peptide_lenght_list:
        possible_peptides += (len(sequence)-lenght+1)
    print("Possible peptides of lenght {} in your sequence ({} amino acids): {}".format(peptide_lenght_list, len(sequence), possible_peptides))
    peptide_set = set()
    for tuple_info in list_of_peptides_cleaved_by_proteasome:
        peptide = tuple_info[0]
        probability = tuple_info[1]
        for lenght in peptide_lenght_list:
            peptide_chosen = peptide[-lenght:]
            peptide_set.add((peptide_chosen, probability))
    #peptide_list = sorted(list(peptide_set))
    print("Predicted proteasome peptides of lenght {}: {}\n".format(peptide_lenght_list, len(peptide_set)))
    print(peptide_set)
    return peptide_set

def proteasome_prediction(sequence, models_export_path, peptide_lenght_list, peptide, linker):
    max_lenght = 7 # change this to 7 if consider using same residue in the middle
    candidate_dictionary = slicing_through_sequence(sequence) #slice
    all_proteasome_predictions = {}
    all_proteasome_predictions_scores = {}
    proteasome_file = "{}/proteasome_all_models.h5".format(models_export_path)
    proteasome_model = load_model(proteasome_file, max_lenght)
    for amino_acid, candidate_position_dictionary in candidate_dictionary.items():
        list_of_candidates = candidate_position_dictionary["candidates"]
        list_of_positions = candidate_position_dictionary["positions"]
        one_hot_df = encode_candidates(list_of_candidates, max_lenght) #one hot encode candidates
        #prediction = proteasome_model.predict_classes(one_hot_df) #make prediction of candidates (0 no cleavage, 1 cleavage)
        prediction = proteasome_model.predict(one_hot_df)
        candidate_df = pd.DataFrame({"sequence": list_of_candidates}) #a dataframe of the predicted sequences
        position_df = pd.DataFrame({"position": list_of_positions}) #a dataframe of the cleavage position (+1) of the predicted sequences
        prediction_df = pd.DataFrame(prediction, columns=["prediction"]) #a dataframe of the predicted cleavages (0 no cleavage, 1 cleavage)
        proteasome_df = pd.concat([candidate_df, position_df, prediction_df], axis=1) #concatenate the above dataframes into a single one
        #print(proteasome_df)
        proteasome_cleavage_regions = proteasome_df.loc[proteasome_df["prediction"] >= 0.5] #select rows with positive proteasome predicted cleavage
        #print(proteasome_cleavage_regions)
        list_of_proteasome_cleavage = proteasome_cleavage_regions["position"].tolist() #a list of the positions of the sequence that are predicted to be cleaved by the proteosome
        list_of_proteasome_cleavage_probabilities = proteasome_cleavage_regions["prediction"].tolist()
        data_probabilities = {}
        for i, pos in enumerate(proteasome_df["position"].tolist()):
            prob = proteasome_df["prediction"].tolist()[i]
            data_probabilities.setdefault(pos, prob)

        #list_of_proteasome_cleavage.extend([0, len(sequence) - 1]) #add initial position and last position of the sequence
        list_of_proteasome_cleavage.sort()
        all_proteasome_predictions.setdefault(amino_acid, list_of_proteasome_cleavage)
        all_proteasome_predictions_scores.update(data_probabilities)       
    
    list_of_peptides_cleaved_by_proteasome, possible_cleavages = peptides_cleaved_by_proteasome(sequence, all_proteasome_predictions, min(peptide_lenght_list), all_proteasome_predictions_scores, peptide, linker)
    peptide_list = hypothetical_peptides(sequence, list_of_peptides_cleaved_by_proteasome, peptide_lenght_list)
    return peptide_list, possible_cleavages
