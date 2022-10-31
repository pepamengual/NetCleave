import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


def score_set(data_path, model_path):
    peptide_lenght = 7
    model = load_model(model_path)
    df = read_data_table(data_path)
    descriptors_df = read_descriptors_table()
    encode_data = encode_sequence_data(df, descriptors_df)
    encoded_df = generate_encoded_df(encode_data, peptide_lenght, descriptors_df)
    prediction = model.predict(encoded_df)
    prediction_df = pd.DataFrame(prediction, columns=["prediction"])
    prediction_df.insert(loc=0, column="sequence", value=df["sequence"])
    
    export_path = "{}_NetCleave.csv".format(data_path.split(".")[0])
    prediction_df.to_csv(export_path, index=False)
    print("Exporting predictions to: {}".format(export_path))
    return prediction_df


def load_model(model_path):
    model_file_path = "{}/{}_model.h5".format(model_path, model_path.split("/")[-1])
    neurons = 336
    model = Sequential()
    model.add(Dense(int(neurons), input_dim=neurons, activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dense(int(neurons/3), activation='tanh', kernel_initializer="glorot_normal"))
    model.add(Dropout(0.1))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.01, momentum=0.00, nesterov=False, name='SGD')
    model.compile(optimizer=opt, loss='binary_crossentropy')
    model.load_weights(model_file_path)
    return model


def read_data_table(path):
    print("---> Reading training data...")
    df = pd.read_csv(path, sep="\t", index_col=None, header=0)
    return df


def read_descriptors_table():
    print("---> Reading descriptors...")
    path = "predictor/ml_main/QSAR_table.csv"
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
    encoded_df = pd.DataFrame(encode_data, columns=["{}_{}".format(i, j) for i in range(peptide_lenght) for j in descriptor_header])
    return encoded_df
