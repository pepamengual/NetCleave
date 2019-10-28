import pickle

def pickle_saver(data, saving_path):
    with open("{}.pickle".format(saving_path), "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
