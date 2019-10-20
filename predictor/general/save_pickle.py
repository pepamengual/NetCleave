import pickle

def pickle_saver(saving_path, data):
    with open(saving_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
