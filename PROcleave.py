import argparse
from predictor.database_functions import ms_extractor
from predictor.database_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.general import read_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import random_peptide_generator
from predictor.core import save_ml_input_data_new
from predictor.ml_main.ml_utilities import read_table
import pandas as pd
from sklearn.model_selection import train_test_split
from predictor.ml_main.ml_utilities import integer_encoding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import gc
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, CuDNNLSTM, GlobalMaxPooling1D
from predictor.ml_main.ml_utilities import plot_history
from predictor.ml_main.ml_utilities import display_model_score
from predictor.ml_main.ml_utilities import metrics_ml
import numpy as np
import dask.dataframe
import datatable
from predictor.ml_main.ml_utilities import array_parser

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --generate_data --features --NN\
"

def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate_data', help='Generate df for ML algorithm.', action='store_true')
    parser.add_argument('--features', help='Add features to df', action='store_true')
    parser.add_argument('--NN', help='Run NN', action='store_true')
    args = parser.parse_args()
    return args.generate_data, args.features, args.NN

def generating_raw_data(iedb_data_file_raw_path, uniprot_data_file_raw_path, n, proteasome_ml_path, erap_ml_path):
    iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
    uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)
    large_uniprot_peptide, list_of_used_peptides = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, n)

    with open("peptides_gbm.txt", "r") as f:
        for line in f:
            peptide = line.rstrip()
            if peptide in list_of_used_peptides:
                print("True")
            else:
                print("False")
    #frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
    frequency_random_model = {'A': 0.08258971312579017, 'C': 0.013826094946210853, 'D': 0.054625650802595425, 'E': 0.0673214708897148, 'F': 0.03866188645338429, 'G': 0.07077863527330625, 'H': 0.022761656446475265, 'I': 0.05923828965491043, 'K': 0.05815460235107699, 'L': 0.09655733034859719, 'M': 0.024154886555486327, 'N': 0.04061129236837406, 'P': 0.047331721936265635, 'Q': 0.03932403048405303, 'R': 0.05534153979141534, 'S': 0.06631318414876945, 'T': 0.05355909368186356, 'V': 0.06865326331945962, 'W': 0.010987143802538912, 'Y': 0.029208513619712422} #speed purposes
    #random_peptides, amino_acid_list, frequency_random_model_list = random_peptide_generator.generate_random_peptides(large_uniprot_peptide, frequency_random_model)
    #save_ml_input_data_new.export_df_for_ml(large_uniprot_peptide, random_peptides, amino_acid_list, frequency_random_model_list, n, proteasome_ml_path, erap_ml_path)

def generating_dataframe_for_NN(proteasome_ml_path, erap_ml_path):
    for path, name in zip([proteasome_ml_path, erap_ml_path], ["proteasome", "erap"]):
        print("Generating {} dataframe".format(name))
        max_length = 10 # for padding, modify it
        training_table = pd.read_csv(path, sep="\t") # Reading dataframes
        sequence_table = training_table.drop(['class'], axis=1) # Getting sequence
        class_table = training_table['class'] # Getting class
        encoding_table = integer_encoding.integer_encoding(sequence_table) # Encoding sequence to integers
        padding_table = pad_sequences(encoding_table, maxlen=max_length, padding='post', truncating='post') # Padding to maximum length
        
        one_hot_table = to_categorical(padding_table, num_classes=20) # One hot encoding
        
        train_ohe = one_hot_table.reshape(sequence_table.shape[0], 1, max_length*20)
        train_ohe = train_ohe.astype(int)
        train_ohe = train_ohe.tolist()
        
        train_ohe_list = []
        for i in train_ohe:
            for j in i:
                train_ohe_list.append(j)
        
        one_hot_df = pd.DataFrame(train_ohe_list)
        training_table = pd.concat([one_hot_df, class_table], axis=1) # Concatenating one hot encoding and class dataframes
        
        exporting_path = "{}/{}/ohe_class_{}.csv".format(path.split("/")[0], path.split("/")[1], name)
        print("Training table of {} built! Writing .csv file training data at {}".format(name, exporting_path))
        training_table.to_csv(exporting_path) # Exporting training data
        
def create_predictive_movels_NN(proteasome_ml_path, erap_ml_path):
    for path, name in zip([proteasome_ml_path, erap_ml_path], ["proteasome", "erap"]):
        if name == "erap":
            print("Creating {} predictive model".format(name))
            training_file_path = "{}/{}/ohe_class_{}.csv".format(path.split("/")[0], path.split("/")[1], name)
            training_table = pd.read_csv(training_file_path, index_col=0)
            data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table.drop(['class'], axis=1), training_table['class'], test_size = 0.20, random_state=42, shuffle=True)
            data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.50, random_state=42, shuffle=True)
            neurons = len(list(training_table.drop(['class'], axis=1)))
            #del training_table
            #gc.collect()

            model = Sequential()
            model.add(Dense(int(neurons*2), input_dim=neurons, activation="sigmoid")) # Hidden Layer 1 that receives the Input from the Input Layer
            model.add(Dense(int(neurons), activation="sigmoid")) # Hidden Layer 2
            model.add(Dense(int(neurons/2), activation="sigmoid")) # Hidden Layer 2
            model.add(Dense(int(neurons/4), activation="sigmoid")) # Hidden Layer 2
            model.add(Dense(1, activation='sigmoid')) #Output layer
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation])
            
            es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=3, verbose=1)
            history1 = model.fit(data_train, class_labels_train, epochs=400, batch_size=256, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
            model.save_weights('model_{}.h5'.format(name))
            plot_history.plot_history(history1)
            display_model_score.display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], 256)


def main(generate_data=False, features=False, NN=False):
    iedb_data_file_raw_path = "../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_data_file_raw_path = "../data/raw/uniprot/uniprot_sprot.fasta"
    iedb_data_file_parsed_path = "../data/parsed/iedb/ms_allele_peptides"
    uniprot_data_file_parsed_path = "../data/parsed/uniprot/uniprot_sequences"
    proteasome_ml_path = "data/NN/proteasome_peptides_class.txt"
    erap_ml_path = "data/NN/erap_peptides_class.txt"
    
    n = 7
 
    if not any([generate_data, features, NN]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")

    if generate_data:
        generating_raw_data(iedb_data_file_raw_path, uniprot_data_file_raw_path, n, proteasome_ml_path, erap_ml_path)

    if features:
        generating_dataframe_for_NN(proteasome_ml_path, erap_ml_path)

    if NN:
        create_predictive_movels_NN(proteasome_ml_path, erap_ml_path)

if __name__ == "__main__":
    generate_data, features, NN = parse_args()
    main(generate_data, features, NN)
