from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from keras import backend as K

def read_data_table(path):
    print("---> Reading training data...")
    training_data_file = "{}/{}_sequence_class.txt".format(path, path.split("/")[-1])
    df = pd.read_csv(training_data_file, sep="\t", index_col=None, header=0)
    sequence_table = df.drop(['class'], axis=1)
    class_table = df['class']
    return sequence_table, class_table

def read_descriptors_table(path):
    print("---> Reading descriptors...")
    df = pd.read_csv(path, sep=",", header=0, index_col=0)
    #scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df

def encode_sequence_data(sequence_table, df):
    print("---> Encoding data using the descriptors...")
    encode_map, encode_data = {}, []
    for r in list("ACDEFGHIKLMNPQRSTVWY"):
        encode_map.setdefault(r, df.loc[r].tolist())

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

def generate_encoded_labeled_df(encoded_df, class_table):#, features_xgboost):
    print("---> Labeling the descriptor dataframe...")
    #scaled_df = scaled_df[features_xgboost]
    encoded_labeled_df = pd.concat([encoded_df, class_table], axis=1)
    encoded_labeled_df = encoded_labeled_df.sample(frac=1).reset_index(drop=True) #shuffles data
    return encoded_labeled_df

def splitting_data(df):
    data_train, data_val, class_labels_train, class_labels_val = train_test_split(df.drop(['class'], axis=1), df['class'], test_size=0.40, random_state=42, shuffle=True)
    data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size=0.25, random_state=42)
    return data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test

def prepare(data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test):
    data_train, data_val, data_test = data_train.to_numpy(), data_val.to_numpy(), data_test.to_numpy()
    data_train, data_val, data_test = np.expand_dims(data_train, axis=0), np.expand_dims(data_val, axis=0), np.expand_dims(data_test, axis=0)
    
    data_train = np.reshape(data_train, (data_train.shape[1], 1, data_train.shape[2]))
    data_val = np.reshape(data_val, (data_val.shape[1], 1, data_val.shape[2]))
    data_test = np.reshape(data_test, (data_test.shape[1], 1, data_test.shape[2]))    

    class_labels_train = class_labels_train.to_numpy().reshape((-1, ))
    class_labels_val = class_labels_val.to_numpy().reshape((-1, ))
    class_labels_test = class_labels_test.to_numpy().reshape((-1, ))
    return data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    tp, tn, fp, fn = K.sum(y_pos * y_pred_pos), K.sum(y_neg * y_pred_neg), K.sum(y_neg * y_pred_pos),  K.sum(y_pos * y_pred_neg)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())

def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def evaluate_models(model, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test, b_size):
    train_score = model.evaluate(data_train, class_labels_train, batch_size=b_size, verbose=1)
    val_score = model.evaluate(data_val, class_labels_val, batch_size=b_size, verbose=1)
    test_score = model.evaluate(data_test, class_labels_test, batch_size=b_size, verbose=1)
    return train_score, val_score, test_score

def run_NN(encoded_labeled_df, models_export_path):
    print("---> NN...")
    b_size = 128
    data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test = splitting_data(encoded_labeled_df)
    data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test = prepare(data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test)
    
    
    neurons = len(list(encoded_labeled_df.drop(['class'], axis=1)))
    print(neurons)
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(int(neurons/3), activation='tanh', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(int(neurons), activation='tanh', kernel_initializer='he_normal'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', matthews_correlation, precision, recall, AUC()])

    es = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1)
    history = model.fit(data_train, class_labels_train, validation_data=(data_val, class_labels_val), epochs=100, batch_size=b_size, callbacks=[es], verbose=1)
    train_score, val_score, test_score = evaluate_models(model, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test, b_size)

    Path(models_export_path).mkdir(parents=True, exist_ok=True)
    model.save_weights("{}/{}_model.h5".format(models_export_path, models_export_path.split("/")[-1]))

def create_models(training_data_path, model_path):
    #features_xgboost = ['2_VHSE6', '0_VHSE6', '1_VHSE6', '0_VHSE5', '1_VHSE7', '1_VHSE8', '0_VHSE1', '0_VHSE3', '3_VHSE3', '0_VHSE7', '2_VHSE7', '1_VHSE3', '3_VHSE2', '3_VHSE5', '1_VHSE2', '2_VHSE8', '1_VHSE5', '2_VHSE2', '1_VHSE1', '3_VHSE1', '2_VHSE5', '2_VHSE1']
    #features_xgboost = ['1_V11', '1_V28', '1_V3', '1_V32', '1_V7', '1_V9', '2_V1', '2_V14', '2_V2', '2_V28', '2_V3', '2_V32', '3_V10', '3_V3', '3_V32']
    #training_data_path = "sample_data.txt"
    print(training_data_path)
    peptide_lenght = 7
    sequence_table, class_table = read_data_table(training_data_path)

    descriptors_path = "predictor/ml_main/QSAR_table.csv"
    #descriptors_path = "QSAR_table.csv"
    df_descriptors = read_descriptors_table(descriptors_path)

    encode_data = encode_sequence_data(sequence_table, df_descriptors)
    encoded_df = generate_encoded_df(encode_data, peptide_lenght, df_descriptors)
    
    encoded_labeled_df = generate_encoded_labeled_df(encoded_df, class_table)#, features_xgboost)
    run_NN(encoded_labeled_df, model_path)

#feature_selection()
