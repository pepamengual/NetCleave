import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, Add, Embedding, Bidirectional, LSTM, Conv1D, GlobalMaxPooling1D
from scripts import plot_history, display_model_score, metrics_ml

def read_input_file(input_file):
    training_table = pd.read_csv(input_file, sep=",")
    sequence_table = training_table.drop(['class'], axis=1)
    class_table = training_table['class']
    return training_table, sequence_table, class_table

def encoder(sequence_table):
    char_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
    char_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 2, 'F': 3, 'G': 0, 'H': 4, 'I': 5, 'K': 6, 'L': 5, 'M': 1, 'N': 7, 'P': 8, 'Q': 7, 'R': 6, 'S': 9, 'T': 9, 'V': 5, 'W': 3, 'Y': 3}   
    encode_list = []
    for row in sequence_table['sequence'].values:
        row_encode = []
        for code in row:
            row_encode.append(char_dict.get(code)) #, 0
        encode_list.append(np.array(row_encode))
    return encode_list, char_dict

def padder(max_lenght, encode_list):
    padding_list = pad_sequences(encode_list, maxlen=max_lenght, padding='post', truncating='post')
    return padding_list

def one_hot_encoder(padding_list, char_dict):
    one_hot_list = to_categorical(padding_list, num_classes=len(set(char_dict.values())))
    return one_hot_list

def reshaper(max_lenght, one_hot_list, sequence_table, char_dict):
    train_ohe = one_hot_list.reshape(sequence_table.shape[0], 1, max_lenght*len(set(char_dict.values())))
    train_ohe = train_ohe.astype(int)
    train_ohe = train_ohe.tolist()
    train_ohe_list = []
    for i in train_ohe:
        for j in i:
            train_ohe_list.append(j)
    one_hot_df = pd.DataFrame(train_ohe_list, columns=["P{}".format(i) for i in range(max_lenght*len(set(char_dict.values())))])
    return one_hot_df

def concatenater(one_hot_df, class_table):
    training_table = pd.concat([one_hot_df, class_table], axis=1)
    return training_table

def splitter(training_table):
    data_train, data_val, class_labels_train, class_labels_val = train_test_split(training_table.drop(['class'], axis=1), training_table['class'], test_size = 0.30, random_state=42, shuffle=True)
    data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val, test_size = 0.25, random_state=42)
    return data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test

def model_generator(training_table, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test):
    neurons = len(list(training_table.drop(['class'], axis=1)))
    model = Sequential()
    model.add(Embedding(len(data_train), neurons, input_length=neurons))
    model.add(Dropout(0.2))
    model.add(Dense(int(neurons), input_dim=neurons, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(LSTM(int(10), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', metrics_ml.matthews_correlation])
    es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=50, verbose=1)
    b_size = 25
    history1 = model.fit(data_train, class_labels_train, epochs=1000, batch_size=b_size, validation_data=(data_val, class_labels_val), callbacks=[es], verbose=1)
    plot_history.plot_history(history1)
    display_model_score.display_model_score(model, [data_train, class_labels_train], [data_val, class_labels_val], [data_test, class_labels_test], b_size)

    train_score = model.evaluate(data_train, class_labels_train, batch_size=b_size, verbose=1)
    val_score = model.evaluate(data_val, class_labels_val, batch_size=b_size, verbose=1)
    test_score = model.evaluate(data_test, class_labels_test, batch_size=b_size, verbose=1)

def main():
    input_file = "all_epitopes_by_class.csv"
    input_file = "negative_and_positive_by_class.csv"
    max_lenght = 9 #nonamers
    training_table, sequence_table, class_table = read_input_file(input_file)
    encode_list, char_dict = encoder(sequence_table)
    padding_list = padder(max_lenght, encode_list)
    one_hot_list = one_hot_encoder(padding_list, char_dict)
    one_hot_df = reshaper(max_lenght, one_hot_list, sequence_table, char_dict)
    training_table = concatenater(one_hot_df, class_table)
    data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test = splitter(training_table)
    model_generator(training_table, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test)
main()
