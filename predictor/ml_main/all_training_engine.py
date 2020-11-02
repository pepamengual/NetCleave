from pathlib import Path
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras import regularizers
from keras import backend as K

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

def padding_sequences(encoding_table, max_lenght):
    padding_table = pad_sequences(encoding_table, maxlen=max_lenght, dtype='int32', padding='post', truncating='post')
    return padding_table

def reshaping_sequences(one_hot_table, sequence_table, max_lenght, column_name):
    train_ohe = one_hot_table.reshape(sequence_table.shape[0], 1, max_lenght*20)
    train_ohe = train_ohe.astype(int)
    train_ohe = train_ohe.tolist()
    train_ohe_list = []
    for i in train_ohe:
        for j in i:
            train_ohe_list.append(j)
    one_hot_df = pd.DataFrame(train_ohe_list, columns=["{}{}".format(column_name, i) for i in range(max_lenght*20)])
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
    max_lenght, b_size = 7, 64 ### MAX LENGHT TO 7 FOR ALL
    resume_prediction = {}
    
    training_table, sequence_table, class_table = read_table(training_data_path)

    """ Encoding sequence into vectors of amino acids
    """
    encoding_table = integer_encoding(sequence_table)
    padding_table = padding_sequences(encoding_table, max_lenght)
    one_hot_table = to_categorical(padding_table, num_classes=20)
    one_hot_df = reshaping_sequences(one_hot_table, sequence_table, max_lenght, column_name="P")
    
    labeled_df = pd.concat([one_hot_df, class_table], axis=1)
    data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test = splitting_data(labeled_df)
    
    neurons = len(list(labeled_df.drop(['class'], axis=1)))
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dense(int(neurons/3), activation='relu', kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(0.001)))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', matthews_correlation, precision, recall, tf.keras.metrics.AUC()])
    
    #es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=5, verbose=1)
    #es = EarlyStopping(monitor='val_auc', mode='max', patience=2, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
    history = model.fit(data_train, class_labels_train, epochs=100, batch_size=b_size, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
    
    Path(models_export_path).mkdir(parents=True, exist_ok=True)
    model.save_weights("{}/{}_model.h5".format(models_export_path, models_export_path.split("/")[-1]))
    
    display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], b_size)
    
    train_score, val_score, test_score = evaluate_models(model, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test, b_size)
    train_dict, val_dict, test_dict = compile_stats(train_score, data_train, val_score, data_val, test_score, data_test)

    resume_prediction.setdefault("Train", train_dict)
    resume_prediction.setdefault("Val", val_dict)
    resume_prediction.setdefault("Test", test_dict)

    resume_df = pd.DataFrame.from_dict(resume_prediction, orient='index')
    #resume_df = pd.DataFrame.from_dict({(i,j): resume_prediction[i][j] for i in resume_prediction.keys() for j in resume_prediction[i].keys()}, orient='index')
    resume_df.to_csv("models_accurary_OHE_val_loss_only_sequence_all_together_decoy_neighbours.csv")

