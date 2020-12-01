from pathlib import Path
import numpy as np
import pandas as pd
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, LSTM, Embedding
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import regularizers
from keras import backend as K
from sklearn import preprocessing


def read_table(training_data_path):
    df_list = []
    training_data_file = "{}/{}_sequence_class.txt".format(training_data_path, training_data_path.split("/")[-1])
    df = pd.read_csv(training_data_file, sep="\t", index_col=None, header=0)
    df_list.append(df)
    training_table = pd.concat(df_list, axis=0, ignore_index=True)
    sequence_table = training_table.drop(['class'], axis=1)
    class_table = training_table['class']
    return training_table, sequence_table, class_table

def integer_encoding(data):
    """
    - Encodes code sequence to integer values.
    - 20 common amino acids are taken into consideration
    """
    print("Reading VHSE data")
    char_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 
                 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 
                 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 
                 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    encode_data = {}
    encode_list = []
    df = pd.read_csv("predictor/ml_main/VHSE_table.csv", sep=",", header=0, index_col=0) #VHSE
    #normalized_df=(df-df.min())/(df.max()-df.min())
    normalized_df = df
    for residue in list("ACDEFGHIKLMNPQRSTVWY"):
        encode_data.setdefault(residue, normalized_df.loc[residue].tolist())

    for row in data['sequence'].values:
        row_encode = []
        for residue in row:
            row_encode.extend(encode_data[residue])
            #row_encode = df.loc[list(row)].stack().tolist()
        encode_list.append(row_encode)
    return encode_list

def generating_data(encode_list, max_lenght, column_name):
    print("Converting into DataFrame")
    one_hot_df = pd.DataFrame(encode_list, columns=["{}{}".format(column_name, i) for i in range(max_lenght*8)]) #8 for VHSE, 48 for QSAR
    return one_hot_df

def splitting_data(labeled_df):
    data_train, data_val, class_labels_train, class_labels_val = train_test_split(labeled_df.drop(['class'], axis=1), labeled_df['class'],
                                                                                                  test_size = 0.40, random_state=42, shuffle=True)
    data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, 
                                                                                test_size = 0.25, random_state=42)
    return data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test 

def display_model_score(model, train, val, test, batch_size):
    train_score = model.evaluate(train[0], train[1], batch_size=batch_size, verbose=1)
    print('Train loss: ', round(train_score[0], 3))
    print('Train accuracy: ', round(train_score[1], 3))
    print('-'*70)
    val_score = model.evaluate(val[0], val[1], batch_size=batch_size, verbose=1)
    print('Val loss: ', round(val_score[0], 3))
    print('Val accuracy: ', round(val_score[1], 3))
    print('-'*70)
    test_score = model.evaluate(test[0], test[1], batch_size=batch_size, verbose=1)
    print('Test loss: ', round(test_score[0], 3))
    print('Test accuracy: ', round(test_score[1], 3))

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

def compile_stats(train_score, data_train, val_score, data_val, test_score, data_test):
    metric_names = ["loss", "accuracy", "mcc", "precision", "recall", "auc"]
    train_dict = {name: round(train_score[i], 3) for i, name in enumerate(metric_names)}
    val_dict = {name: round(val_score[i], 3) for i, name in enumerate(metric_names)}
    test_dict = {name: round(test_score[i], 3) for i, name in enumerate(metric_names)}
    train_dict.setdefault("lenght", len(data_train))
    val_dict.setdefault("lenght", len(data_val))
    test_dict.setdefault("lenght", len(data_test))
    return train_dict, val_dict, test_dict

def create_models(training_data_path, models_export_path):
    max_lenght, b_size = 7, 32 ### MAX LENGHT TO 7 FOR ALL
    resume_prediction = {}
    
    training_table, sequence_table, class_table = read_table(training_data_path)

    """ Encoding sequence into vectors of amino acids
irom sklearn
    """
    encoding_table = integer_encoding(sequence_table)
    one_hot_df = generating_data(encoding_table, max_lenght, column_name="P")
 
    scaler = preprocessing.StandardScaler()
    scaled_data = scaler.fit_transform(one_hot_df)
    scaled_df = pd.DataFrame(scaled_data, columns=one_hot_df.columns)
    print(scaled_df)
    print(scaled_data.mean(axis = 0))
    print(scaled_data.std(axis = 0))
    
    labeled_df = pd.concat([scaled_df, class_table], axis=1)
    data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test = splitting_data(labeled_df)
    data_train, data_val, data_test = data_train.to_numpy(), data_val.to_numpy(), data_test.to_numpy()
    
    data_train, data_val, data_test = np.expand_dims(data_train, axis=0), np.expand_dims(data_val, axis=0), np.expand_dims(data_test, axis=0)
    print(data_train.shape)
    class_labels_train, class_labels_val, class_labels_test = class_labels_train.to_numpy().reshape((-1, )), class_labels_val.to_numpy().reshape((-1, )), class_labels_test.to_numpy().reshape((-1, ))       
    print(class_labels_train.shape)
    data_train = np.reshape(data_train, (data_train.shape[1], 1, data_train.shape[2]))
    data_val = np.reshape(data_val, (data_val.shape[1], 1, data_val.shape[2]))
    data_test = np.reshape(data_test, (data_test.shape[1], 1, data_test.shape[2]))
    print(data_train.shape)
    neurons = len(list(labeled_df.drop(['class'], axis=1)))

    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(1, data_test.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    es = EarlyStopping(monitor='val_auc', mode='max', patience=10, verbose=1)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', matthews_correlation, precision, recall, tf.keras.metrics.AUC()])
    print(model.summary())
    
    history = model.fit(data_train, class_labels_train, validation_data=(data_val, class_labels_val), epochs=100, batch_size=128, callbacks=[es], verbose=1)



    #model.add(Dense(int(neurons), input_dim=neurons, activation='tanh'))
    #model.add(Dense(int(neurons/3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.0005)))
    #model.add(LSTM(100))
    #model.add(Dropout(0.3))
    #model.add(Dense(int(neurons), activation='tanh'))
    #model.add(Dense(1, activation='sigmoid'))
    
    #initial_learning_rate = 0.1
    #lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True)
    
    #model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule), loss='binary_crossentropy', metrics=['accuracy', matthews_correlation, precision, recall, tf.keras.metrics.AUC()])
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', matthews_correlation, precision, recall, tf.keras.metrics.AUC()])
    #print(model.summary())
    #model.compile(optimizer = keras.optimizers.Adam(
    #                lr=0.01,#0.001
    #                beta_1=0.9,
    #                beta_2=0.999,
    #                epsilon=1e-07),#1e-07 default #8
    #                loss='binary_crossentropy',
    #                metrics=['accuracy', matthews_correlation, precision, recall, tf.keras.metrics.AUC()])
    #es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=5, verbose=1)
    #es = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
    #es = EarlyStopping(monitor='val_loss', mode='min', patience=100, verbose=1) #val_loss, min
    #history = model.fit(data_train, class_labels_train, epochs=1000, batch_size=b_size, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
    
    Path(models_export_path).mkdir(parents=True, exist_ok=True)
    model.save_weights("{}/{}_model_VHSE.h5".format(models_export_path, models_export_path.split("/")[-1])) #VHSE
    
    display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], b_size)
    
    train_score, val_score, test_score = evaluate_models(model, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test, b_size)
    train_dict, val_dict, test_dict = compile_stats(train_score, data_train, val_score, data_val, test_score, data_test)

    resume_prediction.setdefault("Train", train_dict)
    resume_prediction.setdefault("Val", val_dict)
    resume_prediction.setdefault("Test", test_dict)

    resume_df = pd.DataFrame.from_dict(resume_prediction, orient='index')
    #resume_df = pd.DataFrame.from_dict({(i,j): resume_prediction[i][j] for i in resume_prediction.keys() for j in resume_prediction[i].keys()}, orient='index')
    resume_df.to_csv("model_VHSE.csv") #VHSE

