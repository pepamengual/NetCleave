"""
Proteasome only
"""
import tensorflow as tf
import argparse
from predictor.database_functions import ms_extractor
from predictor.database_functions import uniprot_extractor
from predictor.general import save_pickle
from predictor.general import read_pickle
from predictor.core import seek_ms_uniprot_and_classify
from predictor.core import non_cleavage_samples
from predictor.core import save_ml_input_data_new
from predictor.ml_main.ml_utilities import read_table
import pandas as pd
from sklearn.model_selection import train_test_split
from predictor.ml_main.ml_utilities import integer_encoding
from keras import regularizers
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.constraints import max_norm
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.layers import Conv1D, Add, MaxPooling1D, BatchNormalization
from keras.layers import Embedding, Bidirectional, LSTM, CuDNNLSTM, GlobalMaxPooling1D
from predictor.ml_main.ml_utilities import plot_history
from predictor.ml_main.ml_utilities import display_model_score
from predictor.ml_main.ml_utilities import metrics_ml
import numpy as np
from predictor.ml_main.ml_utilities import array_parser

HELP = " \
Command:\n \
----------\n \
Run: python3 PROcleave.py --generate_data --features --NN\
"

def parse_args():
    parser = argparse.ArgumentParser(description=HELP, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--generate_data', help='Generate df for ML algorithm.', action='store_true')
    parser.add_argument('--NN', help='Run NN', action='store_true')
    args = parser.parse_args()
    return args.generate_data, args.NN

def generating_raw_data(iedb_data_file_raw_path, uniprot_data_file_raw_path, proteasome_ml_path):
    print("Reading IEDB data")
    iedb_data = ms_extractor.extract_ms_data(iedb_data_file_raw_path)
    
    print("IEDB completed\nExtracting uniprot data")
    uniprot_data = uniprot_extractor.id_sequence_extractor(uniprot_data_file_raw_path)
    
    print("Uniprot data completed\nSeeking neighbour regions")
    large_uniprot_peptide, amino_acid_dict_and_large_uniprot_peptide = seek_ms_uniprot_and_classify.seeking_ms(iedb_data, uniprot_data)

    print("Seeking neightbour completed\nGetting non-cleavage samples")
    non_cleavage_samples.export_df_for_ml(amino_acid_dict_and_large_uniprot_peptide, proteasome_ml_path)

def create_predictive_movels_NN(proteasome_ml_path):
    amino_acids = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
    properties_dict = {}
    properties_df = pd.read_csv("data/normalized_0_1_all.csv", index_col=0)
    for amino_acid in amino_acids:
        physicochemical_properties = [round(i, 3) for i in properties_df.loc[amino_acid].tolist()]
        properties_dict.setdefault(amino_acid, physicochemical_properties)
    print(properties_df)
    resume_prediction = {}
    
    amino_acids = ["A"]
    for amino_acid in amino_acids:
        file_name = "data/NN/proteasome_{}_sequence_class.txt".format(amino_acid)
        print("Working with {}".format(file_name))
        training_table = pd.read_csv(file_name, sep="\t")
        sequence_table = training_table.drop(['class'], axis=1)
        class_table = training_table['class']
        
        print("Encoding {}...".format(amino_acid))
        encoding_table = integer_encoding.integer_encoding(sequence_table)
        max_lenght = 9 #padding
        padding_table = pad_sequences(encoding_table, maxlen=max_lenght, padding='post', truncating='post')
        print("One Hot Encoding {}...".format(amino_acid))
        one_hot_table = to_categorical(padding_table, num_classes=20)
        print("Reshaping {}...".format(amino_acid))
        train_ohe = one_hot_table.reshape(sequence_table.shape[0], 1, max_lenght*20)
        train_ohe = train_ohe.astype(int)
        train_ohe = train_ohe.tolist()
        train_ohe_list = []
        for i in train_ohe:
            for j in i: 
                train_ohe_list.append(j)
        one_hot_df = pd.DataFrame(train_ohe_list, columns=["P{}".format(i) for i in range(9*20)])#7
        print(one_hot_df)
        print("Concatenating {}...".format(amino_acid))
        training_table = pd.concat([one_hot_df, class_table], axis=1)
        #training_table = training_table.drop(["P{}".format(i) for i in range(60,80)], axis=1) # drop data regarding to position 4
        print("Splitting training, validation and testing {}...".format(amino_acid))
        data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table.drop(['class'], axis=1), training_table['class'], test_size = 0.40, random_state=42, shuffle=True)
        data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.25, random_state=42)
        print("Generating model {}...".format(amino_acid))
        
        neurons = len(list(training_table.drop(['class'], axis=1)))
        model = Sequential()
        #model.add(Embedding(len(data_train), neurons, input_length=neurons))
        model.add(Dense(int(neurons), input_dim=neurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001))) # input_dim=neurons
        model.add(Dense(int(neurons/3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))
        #model.add(Dense(int(neurons/3), activation='relu', kernel_initializer='he_normal'))
        #model.add(Bidirectional(LSTM(int(neurons/3), return_sequences=True)))
        #model.add(Dense(int(neurons/4.5), activation='relu', kernel_initializer='he_normal'))
        #model.add(Bidirectional(LSTM(int(neurons/6), return_sequences=True)))
        #model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation]) # tf.keras.metrics.TruePositives()
        es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=5, verbose=1) # val_matthews_correlation
        b_size = 128
        history1 = model.fit(data_train, class_labels_train, epochs=1000, batch_size=b_size, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
        model.save_weights("data/models/proteasome_{}_model.h5".format(amino_acid))
        plot_history.plot_history(history1)
        display_model_score.display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], b_size)
        
        train_score = model.evaluate(data_train, class_labels_train, batch_size=b_size, verbose=1)
        val_score = model.evaluate(data_val, class_labels_val, batch_size=b_size, verbose=1)
        test_score = model.evaluate(data_test, class_labels_test, batch_size=b_size, verbose=1)
        
        resume_prediction.setdefault(amino_acid, {}).setdefault("Train", round(train_score[1], 3))
        resume_prediction.setdefault(amino_acid, {}).setdefault("Val", round(val_score[1], 3))
        resume_prediction.setdefault(amino_acid, {}).setdefault("Test", round(test_score[1], 3))
    
    resume_df = pd.DataFrame.from_dict(resume_prediction, orient='index')
    resume_df.to_csv("models_accurary_8_no_embedding.csv")

def main(generate_data=False, NN=False):
    iedb_data_file_raw_path = "../../data/raw/iedb/mhc_ligand_full.csv"
    uniprot_data_file_raw_path = "../../data/raw/uniprot/uniprot_sprot.fasta"
    iedb_data_file_parsed_path = "../../data/parsed/iedb/ms_allele_peptides"
    uniprot_data_file_parsed_path = "../../data/parsed/uniprot/uniprot_sequences"
    proteasome_ml_path = "data/NN/proteasome"
    
    if not any([generate_data, NN]):
        print("Please, provide an argument. See python3 PROcleave.py -h for more information")

    if generate_data:
        generating_raw_data(iedb_data_file_raw_path, uniprot_data_file_raw_path, proteasome_ml_path)
    
    if NN:
        create_predictive_movels_NN(proteasome_ml_path)

if __name__ == "__main__":
    generate_data, NN = parse_args()
    main(generate_data, NN)
