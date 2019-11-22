import pickle

def pickle_reader(reading_path):
    with open("{}.pickle".format(reading_path), "rb") as f:
        data = pickle.load(f)
        return data
