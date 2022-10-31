import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD


def score_set(data_path, model_path, qsar_table):
    peptide_lenght = 7
    model = load_model(model_path)
    dfs = {}
    if data_path.endswith(".csv"):
        df = read_data_table(data_path)
        dfs.setdefault("csv", df)
    elif data_path.endswith(".fasta") or data_path.endswith(".fst"):
        data_dict = read_fasta_into_dictionary(data_path)
        for name, sequence in data_dict.items():
            peptide_list = [sequence[i:i+peptide_lenght] for i, _ in enumerate(sequence)]
            peptide_list = [i for i in peptide_list if len(i) == peptide_lenght]
            df = pd.DataFrame(peptide_list, columns=["sequence"])
            dfs.setdefault(name, df)

    descriptors_df = read_descriptors_table(qsar_table)
    prediction_dfs = {}
    for name, df in dfs.items():
        encode_data = encode_sequence_data(df, descriptors_df)
        encoded_df = generate_encoded_df(encode_data, peptide_lenght, descriptors_df)
        prediction = model.predict(encoded_df)
        prediction_df = pd.DataFrame(prediction, columns=["prediction"])
        prediction_df.insert(loc=0, column="sequence", value=df["sequence"])

        if name == "csv":
            export_path = "{}_NetCleave.csv".format(data_path.split(".")[0])
        else:
            export_path = "{}_{}_NetCleave.csv".format(data_path.split(".")[0], name)
        prediction_df.to_csv(export_path, index=False)
        prediction_dfs.setdefault(name, prediction_df)
        print("Exporting predictions to: {}".format(export_path))

    if data_path.endswith(".fasta") or data_path.endswith(".fst"):
        plot_path = "{}_NetCleave.png".format(data_path.split(".")[0])
        plot_cleavage_site_frequencies_set(prediction_dfs, plot_path)


def plot_cleavage_site_frequencies_set(prediction_dfs, plot_path):
    fig = plt.figure(figsize=(15, 6))
    for name, df in prediction_dfs.items():
        y_values = list(df["prediction"])
        x_values = [i+3 for i, _ in enumerate(y_values, start=1)]
        plt.plot(x_values, y_values, label=name)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.axhline(y=0.5, color="black", lw=1.5)
    plt.ylabel("NetCleave score")
    plt.xlabel("Cleavage site position in sequence")
    fig.savefig(plot_path, dpi=300)


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


def read_fasta_into_dictionary(fasta_path):
    fasta_dict = {}
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                header = line[1:]
                name = header.split("|")[0]
                fasta_dict.setdefault(name, "")
            else:
                fasta_dict[name] += line
    return fasta_dict


def read_descriptors_table(qsar_table):
    print("---> Reading descriptors...")
    df = pd.read_csv(qsar_table, sep=",", header=0, index_col=0)
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
