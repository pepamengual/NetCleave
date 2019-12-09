import argparse
from predictor.database_functions import ms_extractor
from predictor.database_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.general import read_pickle
from predictor.core import seek_ms_uniprot
from predictor.core import random_model
from predictor.core import random_peptide_generator
from predictor.core import save_ml_input_data
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

HELP = " \
Command:\n \
----------\n \
Run: python3 bidirectional_LSTM.py --generate_data --LSTM\
"

def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate_data', help='Generate df for ML algorithm.', action='store_true')
    parser.add_argument('--features', help='Add features to df', action='store_true')
    parser.add_argument('--NN', help='Run NN', action='store_true')
    args = parser.parse_args()
    return args.generate_data, args.features, args.NN

def aaa(numbers):
    result = []
    for peptido in numbers:
        result1 = [x for b in peptido for x in b]
        result.append(result1)
    result_np = np.array(result)
    return result_np

def main(generate_data=False, features=False, NN=False):
    iedb_data_file_raw_path = "../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_data_file_raw_path = "../data/raw/uniprot/uniprot_sprot.fasta"
    iedb_data_file_parsed_path = "../data/parsed/iedb/ms_allele_peptides"
    uniprot_data_file_parsed_path = "../data/parsed/uniprot/uniprot_sequences"
    n = 5 # 6
    y = 2 # 1
    proteasome_ml_path = "data/LSTM/proteasome_df_LSTM.txt"
    erap_ml_path = "data/LSTM/erap_df_LSTM.txt"
 
    if not any([generate_data, features, NN]):
        print("\nPlease, provide an argument. See python3 bidirectional_LSTM.py -h for more information\n")

    if generate_data:
        print("\n---> Extracting data from IEDB and UNIPROT...\n")
        iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
        uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)
        
        print("\n---> Seeking MS peptides into UNIPROT sequences...\n")
        large_uniprot_peptide = seek_ms_uniprot.seeking_ms(iedb_data, uniprot_data, n)
        
        print("\n---> Generating random peptides...\n")
        #frequency_random_model = random_model.random_model_uniprot_collections(uniprot_data)
        frequency_random_model = {'A': 0.08258971312579017, 'C': 0.013826094946210853, 'D': 0.054625650802595425, 'E': 0.0673214708897148, 'F': 0.03866188645338429, 'G': 0.07077863527330625, 'H': 0.022761656446475265, 'I': 0.05923828965491043, 'K': 0.05815460235107699, 'L': 0.09655733034859719, 'M': 0.024154886555486327, 'N': 0.04061129236837406, 'P': 0.047331721936265635, 'Q': 0.03932403048405303, 'R': 0.05534153979141534, 'S': 0.06631318414876945, 'T': 0.05355909368186356, 'V': 0.06865326331945962, 'W': 0.010987143802538912, 'Y': 0.029208513619712422}
        random_peptides, amino_acid_list, frequency_random_model_list = random_peptide_generator.generate_random_peptides(large_uniprot_peptide, frequency_random_model)
    
        print("\n---> Exporting df for ML algorithms...\n")
        save_ml_input_data.export_df_for_ml(large_uniprot_peptide, random_peptides, amino_acid_list, frequency_random_model_list, n, y, proteasome_ml_path, erap_ml_path)

    if features: # This is to slow (extracting features)...
        print('\n---> Reading df for ML algorithms...\n')
        path = proteasome_ml_path
        training_table = pd.read_csv(path, sep="\t") # index_col=0
        sequence_table = training_table.drop(['class'], axis=1)
        class_table = training_table['class']
        
        encoding_table = integer_encoding.integer_encoding(sequence_table)
        
        max_length = 6
        padding_table = pad_sequences(encoding_table, maxlen=max_length, padding='post', truncating='post')
        
        one_hot_table = to_categorical(padding_table)
        print(one_hot_table[0])
        print(one_hot_table.shape)       

        print("One hot encoding...")
        train_ohe = aaa(one_hot_table)
        #train_ohe = pd.DataFrame(one_hot_table)
        one_hot_df = pd.DataFrame(train_ohe)
        print("Concatenating dataframes...")
        training_table = pd.concat([one_hot_df, class_table], axis=1)
        #training_table = pd.concat([one_hot_df, properties_df, class_table], axis=1)
        print(training_table)
        print(training_table.shape)
        training_table.to_csv("ohe_proteasome.csv")

    if NN:
        print("Reading training table")
        training_table = pd.read_csv("ohe_proteasome.csv", index_col=0)
        
        print("Training table read")
        print(training_table)

        print("\n---> Splitting data into training, validation and testing...\n")
        data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table.drop(['class'], axis=1), training_table['class'], test_size = 0.20, random_state=42, shuffle=True)
        data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.50, random_state=42, shuffle=True)
        neurons = len(list(training_table.drop(['class'], axis=1)))
        del training_table
        gc.collect()

        print("\n---> Constructing the NN...\n")
        model = Sequential()
        model.add(Dense(int(neurons*2), input_dim=neurons, activation="sigmoid")) # Hidden Layer 1 that receives the Input from the Input Layer

        model.add(Dense(int(neurons), activation="sigmoid")) # Hidden Layer 2

        model.add(Dense(int(neurons/2), activation="sigmoid")) # Hidden Layer 2

        model.add(Dense(int(neurons/4), activation="sigmoid")) # Hidden Layer 2

        model.add(Dense(1, activation='sigmoid')) #Output layer
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation])
        """
 
        activation: sigmoid
        activation_untested = softmax
        activation_untested = tanh
        activation_untested = relu
 
        loss: binary_crossentropy
        loss_untested: categorical_crossentropy
 
        metrics: accuracy
        metrics_untested: matthews_correlation
        """
        #es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
        es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=3, verbose=1)
    
        history1 = model.fit(data_train, class_labels_train, epochs=400, batch_size=256, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
        model.save_weights('model_LSTM.h5')
       
        plot_history.plot_history(history1)
        display_model_score.display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], 256)

if __name__ == "__main__":
    generate_data, features, NN = parse_args()
    main(generate_data, features, NN)
