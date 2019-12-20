import pandas as pd

def read_training_table(table_path):
    print("Reading training table {}...".format(table_path))
    training_table = pd.read_table(table_path)
    print("Training table {} read...".format(table_path))
    return training_table

