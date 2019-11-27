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
        training_table = pd.read_csv(path, sep="\t")
        sequence_table = training_table.drop(['class'], axis=1)
        class_table = training_table['class']
        
        encoding_table = integer_encoding.integer_encoding(sequence_table)
        
        max_length = (n - 1) * 2
        padding_table = pad_sequences(encoding_table, maxlen=max_length, padding='post', truncating='post')
        
        one_hot_table = to_categorical(padding_table)
        
        #properties = pd.read_csv("normalized_properties.csv", index_col=0)
        properties = pd.read_csv("aa_index_data/aa_index_normalized.csv", index_col=0)
        
        dict_of_properties = {}
        index_properties_row = properties.axes[0].tolist()
        for amino_acid in index_properties_row:
            property_residue = list(properties.loc[[amino_acid]].values[0])
            dict_of_properties.setdefault(amino_acid, property_residue)

        properties_list = []
        header_list = []
        many_sequences = len(sequence_table["sequence"])
        one_percent = int(many_sequences / 100)
        
        print("Extracting features")
        list_properties = list(properties)
        for i, sequence in enumerate(sequence_table["sequence"]):
            sequence_properties = []
            for residue in sequence:
                sequence_properties.extend(dict_of_properties[residue]) # extend
            #sequence_properties = [dict_of_properties[residue] for residue in sequence]
            #sequence_properties = [properties.loc[[residue]].values[0] for residue in sequence]
            #sequence_properties = [properties.get_value(residue, prop) for residue in sequence for prop in list_properties]
            properties_list.append(sequence_properties)
            if i == 0:
                header = ["P{}".format(i) for i in range(len(sequence_properties))]
                for name in header:
                    header_list.append(name)
            if i % one_percent == 0:
                ratio = round(i / many_sequences * 100, 2)
                print("{}%: {} of {} sequences".format(ratio, i,  many_sequences))

        properties_df = pd.DataFrame(properties_list, columns=header_list)
        print(properties_df)

        print("One hot encoding...")
        train_ohe = aaa(one_hot_table)
        one_hot_df = pd.DataFrame(train_ohe)
        print("Concatenating dataframes...")
        training_table = pd.concat([one_hot_df, properties_df, class_table], axis=1)
        print(training_table)
        print(training_table.shape)
        training_table.to_csv("final_properties_ml_aaindex.csv")

    if NN:
        print("Reading training table")
        features_selected_rfe = ['P2', 'P3', 'P10', 'P18', 'P19', 'P22', 'P24', 'P33', 'P60', 'P69', 'P74', 'P94', 'P123', 'P144', 'P165', 'P173', 'P182', 'P193', 'P219', 'P230', 'P231', 'P233', 'P243', 'P248', 'P249', 'P250', 'P258', 'P271', 'P284', 'P295', 'P301', 'P303', 'P304', 'P307', 'P324', 'P334', 'P346', 'P347', 'P366', 'P373', 'P386', 'P391', 'P404', 'P421', 'P427', 'P450', 'P451', 'P454', 'P494', 'P548', 'P564', 'P565', 'P567', 'P569', 'P598', 'P607', 'P612', 'P613', 'P625', 'P635', 'P654', 'P655', 'P660', 'P662', 'P694', 'P698', 'P711', 'P718', 'P730', 'P764', 'P782', 'P802', 'P810', 'P821', 'P822', 'P826', 'P839', 'P848', 'P852', 'P855', 'P858', 'P863', 'P882', 'P887', 'P897', 'P909', 'P916', 'P923', 'P925', 'P928', 'P931', 'P939', 'P943', 'P956', 'P967', 'P996', 'P1007', 'P1032', 'P1046', 'P1049', 'P1107', 'P1120', 'P1122', 'P1130', 'P1148', 'P1152', 'P1154', 'P1170', 'P1178', 'P1190', 'P1193', 'P1211', 'P1215', 'P1231', 'P1237', 'P1240', 'P1247', 'P1278', 'P1285', 'P1325', 'P1339', 'P1349', 'P1350', 'P1363', 'P1376', 'P1390', 'P1410', 'P1429', 'P1434', 'P1463', 'P1464', 'P1465', 'P1472', 'P1475', 'P1477', 'P1490', 'P1503', 'P1505', 'P1508', 'P1510', 'P1525', 'P1531', 'P1543', 'P1549', 'P1556', 'P1557', 'P1560', 'P1578', 'P1654', 'P1656', 'P1666', 'P1681', 'P1683', 'P1685', 'P1688', 'P1707', 'P1719', 'P1732', 'P1747', 'P1748', 'P1772', 'P1782', 'P1803', 'P1824', 'P1834', 'P1850', 'P1852', 'P1853', 'P1854', 'P1859', 'P1888', 'P1893', 'P1894', 'P1896', 'P1902', 'P1929', 'P1932', 'P1960', 'P1986', 'P1996', 'P2003', 'P2018', 'P2022', 'P2025', 'P2032', 'P2034', 'P2036', 'P2051', 'P2056', 'P2062', 'P2074', 'P2078', 'P2086', 'P2097', 'P2119', 'P2139', 'P2150', 'P2155', 'P2200', 'P2201', 'P2256', 'P2267', 'P2284', 'P2285', 'P2300', 'P2310', 'P2313', 'P2316', 'P2325', 'P2336', 'P2379', 'P2398', 'P2431', 'P2458', 'P2470', 'P2481', 'P2490', 'P2492', 'P2496', 'P2507', 'P2510', 'P2513', 'P2515', 'P2524', 'P2533', 'P2536', 'P2537', 'P2546', 'P2582', 'P2584', 'P2588', 'P2605', 'P2607', 'P2611', 'P2615', 'P2619', 'P2623', 'P2628', 'P2633', 'P2656', 'P2672', 'P2680', 'P2692', 'P2703', 'P2708', 'P2710', 'P2717', 'P2721', 'P2754', 'P2762', 'P2785', 'P2792', 'P2808', 'P2813', 'P2815', 'P2824', 'P2825', 'P2852', 'P2860', 'P2865', 'P2879', 'P2906', 'P2917', 'P2928', 'P2930', 'P2933', 'P2937', 'P2968', 'P2983', 'P3015', 'P3022', 'P3024', 'P3035', 'P3036', 'P3046', 'P3060', 'P3064', 'P3066', 'P3075', 'P3076', 'P3086', 'P3087', 'P3088', 'P3101', 'P3109', 'P3113', 'P3131', 'P3137', 'P3140', 'P3145', 'P3149', 'P3172', 'P3176', 'P3203', 'P3219', 'P3256', 'P3258', 'P3261', 'P3270', 'P3277']
        
        features_selected_rfe = ['P2', 'P33', 'P60', 'P69', 'P94', 'P123', 'P144', 'P233', 'P243', 'P248', 'P250', 'P258', 'P284', 'P295', 'P366', 'P373', 'P421', 'P427', 'P451', 'P454', 'P565', 'P607', 'P612', 'P660', 'P662', 'P694', 'P718', 'P802', 'P810', 'P821', 'P826', 'P852', 'P858', 'P882', 'P916', 'P923', 'P925', 'P996', 'P1032', 'P1049', 'P1107', 'P1120', 'P1154', 'P1170', 'P1178', 'P1231', 'P1247', 'P1278', 'P1350', 'P1363', 'P1376', 'P1390', 'P1410', 'P1429', 'P1463', 'P1464', 'P1490', 'P1531', 'P1654', 'P1656', 'P1688', 'P1707', 'P1732', 'P1803', 'P1824', 'P1850', 'P1854', 'P1859', 'P1893', 'P1894', 'P1902', 'P1960', 'P1996', 'P2022', 'P2025', 'P2032', 'P2078', 'P2119', 'P2155', 'P2201', 'P2284', 'P2300', 'P2336', 'P2398', 'P2431', 'P2458', 'P2470', 'P2481', 'P2490', 'P2507', 'P2588', 'P2605', 'P2619', 'P2623', 'P2633', 'P2656', 'P2692', 'P2703', 'P2708', 'P2717', 'P2808', 'P2824', 'P2860', 'P2865', 'P2917', 'P2928', 'P3015', 'P3035', 'P3036', 'P3088', 'P3109', 'P3113', 'P3137', 'P3140', 'P3145', 'P3149', 'P3203', 'P3219', 'P3256', 'P3261']
        
        features_selected_rfe = ['P60', 'P233', 'P258', 'P366', 'P451', 'P565', 'P694', 'P802', 'P1032', 'P1049', 'P1120', 'P1178', 'P1247', 'P1429', 'P1464', 'P1854', 'P1894', 'P2022', 'P2032', 'P2119', 'P2284', 'P2619', 'P2656', 'P2703', 'P2717', 'P2860', 'P2928', 'P3015', 'P3137', 'P3261']
        
        features_selected_rfe.append("class")
        #training_table = pd.read_csv("final_properties_ml_aaindex.csv", index_col=0)
        training_table = pd.read_csv("final_properties_ml_aaindex.csv", index_col=0, usecols=features_selected_rfe)
        #training_table = dask.dataframe.read_csv("final_properties_ml_aaindex.csv")
        #training_table = datatable.fread("final_properties_ml_aaindex.csv")
        print("Training table read")
        print(training_table)
        #training_table_texts = training_table.drop(['class'], axis=1)
        
        #training_table_texts = training_table_texts.filter(like = 'P', axis = 1) # Starting with P
        #training_table_texts = training_table_texts.drop(training_table_texts.filter(like = 'P', axis = 1).columns, axis = 1) # Not starting with P
        #training_table_texts = training_table_texts.drop(list(training_table_texts.filter(regex = 'P')), axis = 1) #

        print("\n---> Splitting data into training, validation and testing...\n")
        data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table.filter(like = 'P', axis = 1), training_table['class'], test_size = 0.20, random_state=42, shuffle=True)
        data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.50, random_state=42, shuffle=True)
        neurons = len(list(training_table.filter(like = 'P', axis = 1)))
        
        del training_table
        gc.collect()

        print("\n---> Constructing the NN...\n")
        model1 = Sequential()
        model1.add(Dense(neurons, input_dim=neurons, activation="sigmoid"))
        model1.add(Dense(1, activation='sigmoid'))
        model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation])
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

        history1 = model1.fit(data_train, class_labels_train, epochs=400, batch_size=256, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
        model1.save_weights('model_LSTM.h5')
       
        plot_history.plot_history(history1)
        display_model_score.display_model_score(model1, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], 256)

if __name__ == "__main__":
    generate_data, features, NN = parse_args()
    main(generate_data, features, NN)
