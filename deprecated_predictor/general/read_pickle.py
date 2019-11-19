import pickle

def pickle_reader(reading_path, data):
    with open(reading_path, "rb") as f:
        data = pickle.load(f)
