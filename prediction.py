import random
import pandas as pd
from keras.models import load_model
from keras.layers import Input, Dense, Dropout, Flatten, Activation
import numpy as np
from keras.models import Model, Sequential
from predictor.ml_main.ml_utilities import integer_encoding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def load_model(proteasome_file):
    neurons = 200
    model = Sequential()
    model.add(Dense(int(neurons*2), input_dim=neurons, activation="sigmoid")) # Hidden Layer 1 that receives the Input from the Input Layer
    model.add(Dense(int(neurons), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(int(neurons/2), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(int(neurons/4), activation="sigmoid")) # Hidden Layer 2
    model.add(Dense(1, activation='sigmoid')) #Output layer
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.load_weights(proteasome_file)
    return model

def encode_candidates(list_of_candidates):
    encode_df = pd.DataFrame({"sequence": list_of_candidates})
    encoding_table = integer_encoding.integer_encoding(encode_df)
    max_length = 10
    padding_table = pad_sequences(encoding_table, maxlen=max_length, padding='post', truncating='post')
    one_hot_table = to_categorical(padding_table, num_classes=20)
    train_ohe = one_hot_table.reshape(encode_df.shape[0], 1, max_length*20)
    train_ohe = train_ohe.astype(int)
    train_ohe = train_ohe.tolist()
    train_ohe_list = []
    for i in train_ohe:
        for j in i:
            train_ohe_list.append(j)
    one_hot_df = pd.DataFrame(train_ohe_list)
    return one_hot_df

def slicing_through_sequence(sequence, proteasome_model):
    list_of_candidates = []
    for position, amino_acid in enumerate(sequence):
        candidate = sequence[position - 4: position + 6]
        if len(candidate) == 10:
            list_of_candidates.append(candidate)
    one_hot_df = encode_candidates(list_of_candidates)
    
    prediction = proteasome_model.predict_classes(one_hot_df)

    print(prediction)
    print(type(prediction))
    
   #list_of_cleavage.append(len(sequence) - 1)
   #
   #print(list_of_cleavage)
   #for position in list_of_cleavage:
   #    list_of_pre_cleavage = [i for i in list_of_cleavage if i < position]
   #    for elem in list_of_pre_cleavage:
   #        cleaved_by_proteosome = sequence[elem:position+1]
   #        if len(cleaved_by_proteosome) >= 12 and len(cleaved_by_proteosome) <= 20:
   #            print(elem, position, sequence[position], cleaved_by_proteosome)

def main():
    proteasome_file = "model_proteasome.h5"
    proteasome_model = load_model(proteasome_file)
    sequence = "MVGTCHSMAASRSTRVTRSTVGLNGLDESFCGRTLRNRSIAHPEEISSHSQVRSRSPKKRAEPVPTQKGTNNGRTSDVRQQSARDSWVSPRKRRLSSSEKDDLERQALESCERRQAEPAPPVFKNIKRCLRAEATNSSEEDSPVKPDKEPGEHRRIVVDHDADFQGAKRACRCLILDDCEKREVKKVNVSEEGPLNAAVVEEITGYLTVNGVDDSDSAVINCDDCQPDGNTKQNNPGSCVLQEESVAGDGDSETQTSVFCGSRKEDSCIDHFVPCTKSDVQVKLEDHKLVTACLPVERRNQLTAESASGPVSEIQSSLRDSEEEVDVVGDSSASKEQCNENSSNPLDTGSERMPVSGEPELSSILDCVSAQMTSLSEPQEHRYTLRTSPRRAALARSSPTKTTSPYRENGQLEETNLSPQETNTTVSDHVSESPTDPAEVPQDGKVLCCDSENYGSEGLSKPPSEARVNIGHLPSAKESASQHTAEEEDDDPDVYYFESDHVALKHNKDYQRLLQTIAVLEAQRSQAVQDLESLGKHQREALKNPIGFVEKLQKKADIGLPYPQRVVQLPEIMWDQYTNSLGNFEREFKHRKRHTRRVKLVFDKVGLPARPKSPLDPKKDGESLSYSMLPLSDGPEGSHNRPQMIRGRLCDDSKPETFNQLWTVEEQKKLEQLLLKYPPEEVESRRWQKIADELGNRTAKQVASRVQKYFIKLTKAGIPVPGRTPNLYIYSRKSSTSRRQHPLNKHLFKPSTFMTSHEPPVYMDEDDDRSCLHSHMSTAAEEASDEESIPIIYRSLPEYKELLQFKKLKKQKLQQMQAESGFVQHVGFKCDNCGVEPIQGVRWHCQDCPPEMSLDFCDSCSDCPHETDIHKEDHQLEPVYKSETFLDRDYCVSQGTSYSYLDPNYFPANR"
    sequence = "MVGTCHSMAASRSTRVTRSTVGLNGLDESFCGRTLRNRSIAHPEEISSHSQVRS"   
    slicing_through_sequence(sequence, proteasome_model)
main()
