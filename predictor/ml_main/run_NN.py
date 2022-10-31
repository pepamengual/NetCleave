from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as k


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
    encoded_df = pd.DataFrame(encode_data, columns=["{}_{}".format(i, j)
                                                    for i in range(peptide_lenght) for j in descriptor_header])
    return encoded_df


def generate_encoded_labeled_df(encoded_df, class_table):
    print("---> Labeling the descriptor dataframe...")
    encoded_labeled_df = pd.concat([encoded_df, class_table], axis=1)
    encoded_labeled_df = encoded_labeled_df.sample(frac=1).reset_index(drop=True)
    return encoded_labeled_df


def splitting_data(df):
    data_train, data_val, class_labels_train, class_labels_val = train_test_split(df.drop(['class'], axis=1),
                                                                                  df['class'], test_size=0.40,
                                                                                  random_state=42, shuffle=True)
    data_val, data_test, class_labels_val, class_labels_test = train_test_split(data_val, class_labels_val,
                                                                                test_size=0.25, random_state=42)
    return data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test


def prepare(data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test):
    data_train, data_val, data_test = data_train.to_numpy(), data_val.to_numpy(), data_test.to_numpy()
    data_train = np.expand_dims(data_train, axis=0)
    data_val = np.expand_dims(data_val, axis=0)
    data_test = np.expand_dims(data_test, axis=0)

    data_train = np.reshape(data_train, (data_train.shape[1], data_train.shape[2]))
    data_val = np.reshape(data_val, (data_val.shape[1], data_val.shape[2]))
    data_test = np.reshape(data_test, (data_test.shape[1], data_test.shape[2]))

    class_labels_train = class_labels_train.to_numpy().reshape((-1, ))
    class_labels_val = class_labels_val.to_numpy().reshape((-1, ))
    class_labels_test = class_labels_test.to_numpy().reshape((-1, ))
    return data_train, data_val, data_test, class_labels_train, class_labels_val, class_labels_test


def matthews_correlation(y_true, y_pred):
    """
    Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    """
    y_pred_pos = k.round(k.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = k.round(k.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = k.sum(y_pos * y_pred_pos)
    tn = k.sum(y_neg * y_pred_neg)
    fp = k.sum(y_neg * y_pred_pos)
    fn = k.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = k.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + k.epsilon())


def precision(y_true, y_pred):
    """
    Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision_value = true_positives / (predicted_positives + k.epsilon())
    return precision_value


def recall(y_true, y_pred):
    """
    Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall_value = true_positives / (possible_positives + k.epsilon())
    return recall_value


def evaluate_models(model, data_train, class_labels_train, data_val, class_labels_val, data_test, class_labels_test, s):
    train_score = model.evaluate(data_train, class_labels_train, batch_size=s, verbose=1)
    val_score = model.evaluate(data_val, class_labels_val, batch_size=s, verbose=1)
    test_score = model.evaluate(data_test, class_labels_test, batch_size=s, verbose=1)
    return train_score, val_score, test_score


def plot_loss(history, models_export_path, path):
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Test')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{models_export_path}/model_{path}.png", dpi=300)


def run_netcleave_neural_network(encoded_labeled_df):
    print("---> NN...")
    b_size = 64
    data_train, data_val, data_test, class_train, class_val, class_test = splitting_data(encoded_labeled_df)
    data_train, data_val, data_test, class_train, class_val, class_test = prepare(data_train, data_val, data_test,
                                                                                  class_train, class_val, class_test)

    neurons = len(list(encoded_labeled_df.drop(['class'], axis=1)))
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dense(int(neurons/3), activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.01, momentum=0.00, nesterov=False, name='SGD')
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy', matthews_correlation, precision, recall, AUC()])

    es = EarlyStopping(monitor='val_matthews_correlation', mode='max', patience=7, verbose=1)
    history = model.fit(data_train, class_train,
                        validation_data=(data_val, class_val),
                        epochs=5000, batch_size=b_size, callbacks=[es], verbose=1)
    _train_score, _val_score, _test_score = evaluate_models(model, data_train, class_train, data_val,
                                                            class_val, data_test, class_test, b_size)
    return model, history


def create_models(training_data_path, model_path, qsar_table):
    print(training_data_path)
    peptide_lenght = 7
    sequence_table, class_table = read_data_table(training_data_path)

    df_descriptors = read_descriptors_table(qsar_table)

    encode_data = encode_sequence_data(sequence_table, df_descriptors)
    encoded_df = generate_encoded_df(encode_data, peptide_lenght, df_descriptors)
    
    encoded_labeled_df = generate_encoded_labeled_df(encoded_df, class_table)
    model, history = run_netcleave_neural_network(encoded_labeled_df)
    filename_tag = training_data_path.split("/")[-1]

    Path(model_path).mkdir(parents=True, exist_ok=True)
    model.save_weights(f"{model_path}/{filename_tag}_model.h5")
    plot_loss(history, model_path, filename_tag)
