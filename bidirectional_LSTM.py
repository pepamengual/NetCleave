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
from sklearn.utils import shuffle
import numpy as np

HELP = " \
Command:\n \
----------\n \
Run: python3 bidirectional_LSTM.py --generate_data --LSTM\
"

def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate_data', help='Generate df for ML algorithm.', action='store_true')
    parser.add_argument('--LSTM_model', help='Running LSTM on data', action='store_true')
    args = parser.parse_args()
    return args.generate_data, args.LSTM_model

def aaa(numbers):
    result = []
    for peptido in numbers:
        result1 = [x for b in peptido for x in b]
        result.append(result1)
    result_np = np.array(result)
    '''
    a = []
    for k in numbers:
        v = []
        for i in k:
            v.append(i)
        v = np.array(v)
        a.append(v)

    for k in numbers:
        len_k = len(k)
        v = np.concatenate(k[0]
    '''
    print(result_np.shape)
    return result_np

def main(generate_data=False, LSTM_model=False):
    iedb_data_file_raw_path = "../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_data_file_raw_path = "../data/raw/uniprot/uniprot_sprot.fasta"
    iedb_data_file_parsed_path = "../data/parsed/iedb/ms_allele_peptides"
    uniprot_data_file_parsed_path = "../data/parsed/uniprot/uniprot_sequences"
    n = 4 # 6
    y = 1 # 1
    proteasome_ml_path = "data/LSTM/proteasome_df_LSTM.txt"
    erap_ml_path = "data/LSTM/erap_df_LSTM.txt"
 
    if not any([generate_data, LSTM_model]):
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

    if LSTM_model:
        print('\n---> Reading df for ML algorithms...\n')
        path = proteasome_ml_path
        training_table = read_table.read_training_table(path)
        training_table
        training_table = shuffle(training_table)
        class_labels = training_table['class']
        training_table_texts = training_table.drop(['class'], axis=1)
        
        print("\n---> Splitting data into training, validation and testing...\n")
        data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table_texts, class_labels, test_size = 0.20, random_state=42)
        data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.50, random_state=42)
    
        print("\n---> Encoding sequences to integers...\n")
        train_encode = integer_encoding.integer_encoding(data_train)
        val_encode = integer_encoding.integer_encoding(data_val)
        test_encode = integer_encoding.integer_encoding(data_test)

        print("\n---> Padding sequences...\n")
        max_length = (n - 1) * 2
        train_pad = pad_sequences(train_encode, maxlen=max_length, padding='post', truncating='post')
        val_pad = pad_sequences(val_encode, maxlen=max_length, padding='post', truncating='post')
        test_pad = pad_sequences(test_encode, maxlen=max_length, padding='post', truncating='post')
        print(type(train_pad))

        print("\n---> One hot encoding...\n")
        train_ohe = to_categorical(train_pad)
        val_ohe = to_categorical(val_pad)
        test_ohe = to_categorical(test_pad)
        
        train_ohe = aaa(train_ohe)
        val_ohe = aaa(val_ohe)
        test_ohe = aaa(test_ohe)
        
        print(train_ohe.shape)
        
        #one = train_ohe[0]
        #print(type(one))
        #print(one)
        
        #del train_encode, val_encode, test_encode
        #gc.collect()

        #print("\n---> Label encoding output variable...\n")
        #le = LabelEncoder()
        #y_train_le = le.fit_transform(class_labels_train)
        #y_val_le = le.transform(class_labels_val)
        #y_test_le = le.transform(class_labels_test)

        #print("\n---> One hot encoding of outputs...\n")
        #y_train = to_categorical(y_train_le)
        #y_val = to_categorical(y_val_le)
        #y_test = to_categorical(y_test_le)
        #print(y_train[0])

        print("\n---> Constructing the Bidirectional LSTM...\n")
        model1 = Sequential()
        model1.add(Dense(126, input_dim=126, activation="sigmoid"))
        print("I")
        model1.add(Dense(1, activation='sigmoid'))
        print("O")
        
        #model1.add(Embedding(21, 128, input_length=max_length))
        #model1.add(Bidirectional(LSTM(128))) # 64
        #model1.add(Dropout(0.5))
        #model1.add(Dense(1, activation='sigmoid')) # 2 
        
        model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation])
        print("C")
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
        es = EarlyStopping(monitor='val_loss', patience=3, verbose=1)

        history1 = model1.fit(train_ohe, class_labels_train, epochs=400, batch_size=256, validation_data=(val_ohe, class_labels_val), callbacks=[es], verbose=1)
        #history1 = model1.fit(train_pad, class_labels_train, epochs=400, batch_size=256, validation_data=(val_pad, class_labels_val), callbacks=[es])
        model1.save_weights('model_LSTM.h5')
        
        plot_history.plot_history(history1)
        display_model_score.display_model_score(model1, [train_ohe, class_labels_train], [val_ohe, class_labels_val], [test_ohe, class_labels_test], 256)
        #display_model_score.display_model_score(model1, [train_pad, class_labels_train], [val_pad, class_labels_val], [test_pad, class_labels_test], 256)

if __name__ == "__main__":
    generate_data, LSTM_model = parse_args()
    main(generate_data, LSTM_model)
